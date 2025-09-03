import os
import psycopg2
import psycopg2.extras
import netCDF4 as nc
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
POSTGRES_CONN_STRING = "dbname='postgres' user='postgres' password='your_password' host='localhost'"
DATA_DIR = 'test_data/'

def process_argo_file(file_path):
    """
    Process Argo file with fallback to different pressure variables
    and better handling of masked data
    """
    all_profiles_data = []
    
    try:
        with nc.Dataset(file_path, 'r') as ds:
            if 'N_PROF' not in ds.dimensions:
                print(f"❌ No N_PROF dimension in {os.path.basename(file_path)}")
                return []
            
            num_profiles = len(ds.dimensions['N_PROF'])
            print(f"Processing {num_profiles} profiles from {os.path.basename(file_path)}")
            
            # Try different pressure variable names in order of preference
            pressure_vars = ['PRES_ADJUSTED', 'PRES']
            temp_vars = ['TEMP_ADJUSTED', 'TEMP']
            salinity_vars = ['PSAL_ADJUSTED', 'PSAL']
            
            # Find available variables
            pressure_var = None
            temp_var = None
            salinity_var = None
            
            for var in pressure_vars:
                if var in ds.variables:
                    pressure_var = var
                    break
                    
            for var in temp_vars:
                if var in ds.variables:
                    temp_var = var
                    break
                    
            for var in salinity_vars:
                if var in ds.variables:
                    salinity_var = var
                    break
            
            if not pressure_var:
                print(f"❌ No pressure variables found in {os.path.basename(file_path)}")
                print(f"Available variables: {list(ds.variables.keys())}")
                return []
            
            print(f"Using: PRES={pressure_var}, TEMP={temp_var}, SAL={salinity_var}")
            
            total_measurements = 0
            
            for i in range(num_profiles):
                try:
                    # Extract basic profile info
                    platform_raw = ds.variables['PLATFORM_NUMBER'][i]
                    platform_id = ''.join([b.decode('utf-8') for b in platform_raw.data if b != b' ']).strip()
                    
                    # Reference date
                    ref_date_raw = ds.variables['REFERENCE_DATE_TIME'][:]
                    ref_date_str = ''.join([b.decode('utf-8') for b in ref_date_raw.data]).strip()
                    base_date = pd.to_datetime(ref_date_str, format='%Y%m%d%H%M%S')
                    
                    # Measurement date
                    juld_value = ds.variables['JULD'][i]
                    if np.ma.is_masked(juld_value) or np.isnan(juld_value):
                        print(f"  Profile {i}: Invalid JULD, skipping")
                        continue
                    
                    measurement_date = base_date + pd.to_timedelta(float(juld_value), unit='D')
                    
                    # Coordinates
                    latitude = float(ds.variables['LATITUDE'][i])
                    longitude = float(ds.variables['LONGITUDE'][i])
                    
                    if np.isnan(latitude) or np.isnan(longitude):
                        print(f"  Profile {i}: Invalid coordinates, skipping")
                        continue
                    
                    # Load measurement arrays
                    pressure = ds.variables[pressure_var][i, :]
                    temperature = ds.variables[temp_var][i, :] if temp_var else np.ma.masked_all_like(pressure)
                    salinity = ds.variables[salinity_var][i, :] if salinity_var else np.ma.masked_all_like(pressure)
                    
                    # Check if we have ANY valid pressure data
                    valid_pressure_mask = ~pressure.mask if hasattr(pressure, 'mask') else ~np.isnan(pressure)
                    valid_count = np.sum(valid_pressure_mask)
                    
                    print(f"  Profile {i}: {valid_count} valid pressure measurements out of {len(pressure)}")
                    
                    if valid_count == 0:
                        print(f"  Profile {i}: No valid measurements, skipping")
                        continue
                    
                    # Extract valid measurements
                    profile_measurements = 0
                    for j in range(len(pressure)):
                        # Check if pressure is valid
                        if hasattr(pressure, 'mask') and pressure.mask[j]:
                            continue
                        if np.isnan(pressure[j]):
                            continue
                        
                        pressure_val = float(pressure[j])
                        if pressure_val < 0:  # Pressure should be positive
                            continue
                        
                        # Temperature (can be None)
                        if temp_var and hasattr(temperature, 'mask') and not temperature.mask[j] and not np.isnan(temperature[j]):
                            temp_val = float(temperature[j])
                        else:
                            temp_val = None
                        
                        # Salinity (can be None)
                        if salinity_var and hasattr(salinity, 'mask') and not salinity.mask[j] and not np.isnan(salinity[j]):
                            sal_val = float(salinity[j])
                        else:
                            sal_val = None
                        
                        all_profiles_data.append((
                            platform_id,
                            measurement_date,
                            latitude,
                            longitude,
                            pressure_val,
                            temp_val,
                            sal_val
                        ))
                        profile_measurements += 1
                    
                    print(f"  Profile {i}: Extracted {profile_measurements} measurements")
                    total_measurements += profile_measurements
                    
                except Exception as e:
                    print(f"  Profile {i}: Error - {e}")
                    continue
            
            print(f"Total measurements extracted: {total_measurements}")
            
            if total_measurements == 0:
                print(f"⚠️  {os.path.basename(file_path)} contains no valid measurements")
                print("This could mean:")
                print("  - All data is flagged as bad quality")
                print("  - The profile is empty/test data")
                print("  - Different variable names are used")
            
        return all_profiles_data
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return []

def main():
    """Main ingestion with better error reporting"""
    pg_conn = None
    try:
        # Check directory
        if not os.path.exists(DATA_DIR):
            print(f"❌ Directory '{DATA_DIR}' not found")
            return
        
        nc_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.nc')]
        if not nc_files:
            print(f"❌ No .nc files in '{DATA_DIR}'")
            return
        
        print(f"Found {len(nc_files)} NetCDF files")
        
        # Connect to database
        pg_conn = psycopg2.connect(POSTGRES_CONN_STRING)
        cursor = pg_conn.cursor()
        print("✓ Connected to PostgreSQL")
        
        total_rows = 0
        files_with_data = 0
        files_without_data = 0
        
        for file_path in nc_files:
            print(f"\n{'='*60}")
            print(f"Processing: {os.path.basename(file_path)}")
            print('='*60)
            
            try:
                data = process_argo_file(file_path)
                
                if not data:
                    print(f"❌ No data from {os.path.basename(file_path)}")
                    files_without_data += 1
                    continue
                
                # Insert data
                print(f"Inserting {len(data)} rows...")
                psycopg2.extras.execute_values(
                    cursor,
                    """INSERT INTO argo_measurements 
                       (platform_id, measurement_date, latitude, longitude, 
                        pressure_dbar, temperature_celsius, salinity_psu) 
                       VALUES %s""",
                    data
                )
                pg_conn.commit()
                
                total_rows += len(data)
                files_with_data += 1
                print(f"✅ Inserted {len(data)} rows")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                pg_conn.rollback()
        
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Files processed: {len(nc_files)}")
        print(f"Files with data: {files_with_data}")
        print(f"Files without data: {files_without_data}")
        print(f"Total rows inserted: {total_rows:,}")
        
        # Verify database
        cursor.execute("SELECT COUNT(*) FROM argo_measurements")
        db_count = cursor.fetchone()[0]
        print(f"Database verification: {db_count:,} rows")
        
    except Exception as e:
        print(f"❌ Main error: {e}")
    finally:
        if pg_conn:
            cursor.close()
            pg_conn.close()

if __name__ == "__main__":
    main()
