import os
import xarray as xr
import pandas as pd
import numpy as np
from pyproj import CRS, Transformer
from datetime import datetime, timedelta
import pytz
import goes2go
from sqlalchemy import create_engine, text
from concurrent.futures import ProcessPoolExecutor
import time

def extract_datetime_from_filename(filename):
    """Extrai data e hora do nome do arquivo no formato padrão GOES."""
    datetime_str = filename.split('_s')[1].split('_')[0]
    date = datetime(int(datetime_str[:4]), 1, 1) + timedelta(days=int(datetime_str[4:7]) - 1)
    return datetime(date.year, date.month, date.day,
                    int(datetime_str[7:9]), int(datetime_str[9:11]), int(datetime_str[11:13]))

def get_last_downloaded_file_time(save_dir):
    """Obtém o timestamp do último arquivo processado no diretório."""
    files = [os.path.join(root, f)
             for root, _, files in os.walk(save_dir)
             for f in files if f.endswith(".nc")]
    return max((extract_datetime_from_filename(os.path.basename(f))
               for f in files), default=None) if files else None

def is_valid_netcdf(file_path):
    """Verifica integridade do arquivo NetCDF."""
    try:
        with xr.open_dataset(file_path) as ds:
            return bool(ds.variables)
    except Exception as e:
        print(f"Arquivo corrompido: {file_path} - Erro: {e}")
        return False

def process_goes_fire_data(input_file):
    """Processa arquivo GOES e extrai dados de focos de calor válidos."""
    max_attempts = 3
    attempt = 0
    while attempt < max_attempts:
        try:
            if not is_valid_netcdf(input_file):
                return pd.DataFrame()

            print(f"Processando: {os.path.basename(input_file)}")
            with xr.open_dataset(input_file) as ds:
                proj = ds.goes_imager_projection
                crs_params = {
                    'grid_mapping_name': 'geostationary',
                    'perspective_point_height': proj.perspective_point_height.item(),
                    'semi_major_axis': proj.semi_major_axis.item(),
                    'semi_minor_axis': proj.semi_minor_axis.item(),
                    'longitude_of_projection_origin': proj.longitude_of_projection_origin.item(),
                    'sweep_angle_axis': proj.sweep_angle_axis
                }

                x = ds.x.values * crs_params['perspective_point_height']
                y = ds.y.values * crs_params['perspective_point_height']
                xx, yy = np.meshgrid(x, y)

                transformer = Transformer.from_crs(CRS.from_cf(crs_params), "EPSG:4326", always_xy=True)
                lon, lat = transformer.transform(xx, yy)

                transformer_4674 = Transformer.from_crs("EPSG:4326", "EPSG:4674", always_xy=True)
                lon_4674, lat_4674 = transformer_4674.transform(lon, lat)

                valid_fires = np.isin(ds['Mask'], [10, 11, 30, 31]) & (ds['DQF'] == 0)
                if 320 < np.nanmin(ds['Temp']) < 400:
                    valid_fires &= (ds['Temp'] > 300)

                y_idx, x_idx = np.where(valid_fires)
                regiao_mask = (
                    (lat_4674[y_idx, x_idx] >= -55) & (lat_4674[y_idx, x_idx] <= 13) &
                    (lon_4674[y_idx, x_idx] >= -85) & (lon_4674[y_idx, x_idx] <= -30)
                )
                y_idx, x_idx = y_idx[regiao_mask], x_idx[regiao_mask]

                file_dt = extract_datetime_from_filename(os.path.basename(input_file)).replace(tzinfo=pytz.utc)
                file_dt_br = file_dt.astimezone(pytz.timezone('America/Sao_Paulo'))

                df = pd.DataFrame({
                    'temp_kelvin': np.round(ds['Temp'].values[y_idx, x_idx], 2),
                    'area_m2': np.round(ds['Area'].values[y_idx, x_idx], 2),
                    'power_mw': np.round(ds['Power'].values[y_idx, x_idx], 2),
                    'file_datetime': file_dt_br,
                    'geom': [f"SRID=4674;POINT({lon_4674[i,j]} {lat_4674[i,j]})"
                            if np.isfinite(lon_4674[i,j]) and np.isfinite(lat_4674[i,j])
                            else None
                            for i, j in zip(y_idx, x_idx)]
                })

                print(f"Arquivo processado: {len(df)} registros válidos.")
                return df
        except Exception as e:
            print(f"Erro no processamento (tentativa {attempt + 1}/{max_attempts}): {e}")
            attempt += 1
            time.sleep(2 ** attempt)
    return pd.DataFrame()

def insert_into_postgis(df, engine):
    """Insere dados no PostGIS com verificação de duplicidades."""
    if df.empty:
        return

    df['dt_obtencao'] = datetime.now(pytz.timezone('America/Sao_Paulo'))

    with engine.connect() as conn:
        for _, row in df.iterrows():
            exists = conn.execute(text("""
                SELECT 1 FROM schema.goes 
                WHERE geom = :geom AND file_datetime = :file_datetime
                """), row[['geom', 'file_datetime']].to_dict()).scalar()

            if not exists:
                row['id'] = conn.execute(text("SELECT nextval('schema.goes_id_seq')")).scalar()
                row.to_frame().T.to_sql('goes', engine, schema='schema',
                                      if_exists='append', index=False)
                print(f"Novo registro inserido ID {row['id']}")
            else:
                print(f"Registro duplicado ignorado: {row['file_datetime']}")

def download_goes_files(save_dir, start_date, end_date):
    """Gerencia o download de arquivos GOES dentro do período especificado."""
    try:
        result = goes2go.goes_timerange(
            satellite='G16',
            product='ABI-L2-FDCF',
            start=start_date,
            end=end_date,
            domain='F',
            download=True,
            return_as='filelist',
            save_dir=save_dir,
            overwrite=False
        )

        files = []
        if isinstance(result, pd.DataFrame):
            files = result['file'].tolist() if not result.empty else []
        elif isinstance(result, list):
            files = result
        else:
            print("Formato de retorno inesperado da API goes2go")
            return []

        valid_files = []
        for f in files:
            abs_path = os.path.abspath(os.path.join(save_dir, f))
            if is_valid_netcdf(abs_path):
                valid_files.append(abs_path)
            else:
                print(f"Removendo arquivo corrompido: {abs_path}")
                os.remove(abs_path)

        return valid_files

    except Exception as e:
        print(f"Falha no download: {str(e)}")
        return []

def process_files_multiprocess(files):
    """Otimiza o processamento utilizando multiprocessamento."""
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = [df for df in executor.map(process_goes_fire_data, files) if not df.empty]
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def main():
    engine = create_engine("postgresql://usuário:Senha@host:port/banco de dados")
    save_dir = "Endereço do diretório de download"

    last_file_time = get_last_downloaded_file_time(save_dir)
    start_date = last_file_time + timedelta(seconds=1) if last_file_time else datetime(2025, 2, 1, 0, 0) #Defina a data de início manualmente (formato: ano, mês, dia, hora, minuto)

    now = datetime.now(pytz.utc)
    end_date = datetime(now.year, now.month, now.day, now.hour, now.minute, now.second) #Define a data final de aquisição para a hora e data atual do sistema

    if start_date > end_date:
        print("Ajustando data inicial para período válido.")
        start_date = end_date - timedelta(days=1)

    arquivos = download_goes_files(save_dir, start_date, end_date)

    if not arquivos:
        print("Nenhum novo arquivo para processar.")
        return

    print(f"Iniciando processamento de {len(arquivos)} arquivos...")
    df = process_files_multiprocess(arquivos)

    if not df.empty:
        insert_into_postgis(df, engine)
        print("Dados inseridos com sucesso.")
    else:
        print("Nenhum dado válido encontrado.")

if __name__ == "__main__":
    main()
