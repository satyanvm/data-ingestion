import os
import psycopg2
import psycopg2.extras
import netCDF4 as nc
import numpy as np
import pandas as pd
from tqdm import tqdm
import re


# --- Configuration ---
POSTGRES_CONN_STRING = "dbname='postgres' user='postgres' password='your_password' host='localhost'"
DATA_DIR = 'test_data/'


def find_variable_case_insensitive(target_vars, available_vars):
    """Find variable names case-insensitively"""
    for target in target_vars:
        for available in available_vars:
            if available.lower() == target.lower():
                return available
    return None


def mask_check(array, idx):
    """
    Safely check if an array element is masked, handling both scalar and array cases.
    This prevents the 'truth value is ambiguous' error.
    """
    if hasattr(array, 'mask'):
        try:
            mask_element = array.mask[idx]
            if isinstance(mask_element, (np.ndarray, np.ma.MaskedArray)):
                return mask_element.any()  # Use .any() for arrays
            else:
                return bool(mask_element)  # Convert to bool for scalars
        except:
            return False
    return False


def safe_isnan(value):
    """
    Safely check if a value contains NaN, handling scalars, arrays, and masked arrays.
    This prevents the 'truth value is ambiguous' error.
    """
    try:
        if isinstance(value, (np.ndarray, np.ma.MaskedArray)):
            if isinstance(value, np.ma.MaskedArray):
                # For masked arrays, consider masked values as NaN
                return np.all(np.logical_or(value.mask, np.isnan(value.data)))
            else:
                return np.all(np.isnan(value))
        else:
            # Scalar case
            return np.isnan(value)
    except:
        return False


def safe_float_conversion(value):
    """
    Safely convert a value to float, handling arrays that should be scalars.
    """
    try:
        if isinstance(value, (np.ndarray, np.ma.MaskedArray)):
            if value.size == 1:
                return float(value.flat[0])  # Get the single element
            else:
                # If it's truly an array, take the first valid element
                if isinstance(value, np.ma.MaskedArray):
                    valid_indices = ~value.mask
                    if np.any(valid_indices):
                        return float(value.data[valid_indices][0])
                else:
                    valid_indices = ~np.isnan(value)
                    if np.any(valid_indices):
                        return float(value[valid_indices][0])
                raise ValueError("No valid values to convert")
        else:
            return float(value)
    except:
        raise ValueError(f"Cannot convert {value} to float")


def process_argo_file(file_path):
    """
    Process Argo file with comprehensive error handling and safe array operations.
    Supports both multi-profile files (with N_PROF) and single-profile files (without N_PROF).
    Handles both uppercase and lowercase variable names for ALL variables.
    Includes safe array checking to prevent all truth value ambiguity errors.
    """
    all_profiles_data = []
    
    try:
        with nc.Dataset(file_path, 'r') as ds:
            # Determine file type and number of profiles
            if 'N_PROF' in ds.dimensions:
                num_profiles = len(ds.dimensions['N_PROF'])
                is_multi_profile = True
                print(f"Multi-profile file: {num_profiles} profiles from {os.path.basename(file_path)}")
            else:
                num_profiles = 1
                is_multi_profile = False
                print(f"Single-profile file from {os.path.basename(file_path)}")
            
            # Get all available variables
            available_vars = list(ds.variables.keys())
            
            # Find all required variables with case-insensitive matching
            pressure_var = find_variable_case_insensitive(['PRES_ADJUSTED', 'PRES'], available_vars)
            temp_var = find_variable_case_insensitive(['TEMP_ADJUSTED', 'TEMP'], available_vars)
            salinity_var = find_variable_case_insensitive(['PSAL_ADJUSTED', 'PSAL'], available_vars)
            
            # Required variables with fallback options
            platform_var_name = find_variable_case_insensitive(
                ['PLATFORM_NUMBER', 'FLOAT_SERIAL_NO', 'WMO_INST_TYPE'], 
                available_vars
            )
            juld_var_name = find_variable_case_insensitive(['JULD'], available_vars)
            lat_var_name = find_variable_case_insensitive(['LATITUDE'], available_vars)
            lon_var_name = find_variable_case_insensitive(['LONGITUDE'], available_vars)
            ref_date_var_name = find_variable_case_insensitive(['REFERENCE_DATE_TIME'], available_vars)
            
            # Check if we have minimum required variables
            missing_vars = []
            if not pressure_var:
                missing_vars.append('pressure (PRES_ADJUSTED/PRES)')
            if not platform_var_name:
                missing_vars.append('platform_number (PLATFORM_NUMBER/FLOAT_SERIAL_NO)')
            if not juld_var_name:
                missing_vars.append('date (JULD)')
            if not lat_var_name:
                missing_vars.append('latitude (LATITUDE)')
            if not lon_var_name:
                missing_vars.append('longitude (LONGITUDE)')
            if not ref_date_var_name:
                missing_vars.append('reference_date (REFERENCE_DATE_TIME)')
            
            if missing_vars:
                print(f"❌ Missing required variables in {os.path.basename(file_path)}: {missing_vars}")
                print(f"Available variables: {available_vars}")
                return []
            
            print(f"Using: PRES={pressure_var}, TEMP={temp_var}, SAL={salinity_var}")
            print(f"       PLATFORM={platform_var_name}, JULD={juld_var_name}")
            print(f"       LAT={lat_var_name}, LON={lon_var_name}, REF_DATE={ref_date_var_name}")
            
            total_measurements = 0
            
            for i in range(num_profiles):
                try:
                    # Extract basic profile info - handle indexing based on file type
                    if is_multi_profile:
                        # Multi-profile: use profile index
                        platform_raw = ds.variables[platform_var_name][i]
                        juld_value = ds.variables[juld_var_name][i]
                        latitude = float(ds.variables[lat_var_name][i])
                        longitude = float(ds.variables[lon_var_name][i])
                        
                        # Load measurement arrays with profile index
                        pressure = ds.variables[pressure_var][i, :]
                        temperature = ds.variables[temp_var][i, :] if temp_var else np.ma.masked_all_like(pressure)
                        salinity = ds.variables[salinity_var][i, :] if salinity_var else np.ma.masked_all_like(pressure)
                    else:
                        # Single-profile: handle scalar and array variables appropriately
                        
                        # Platform ID
                        platform_var = ds.variables[platform_var_name]
                        if len(platform_var.shape) == 0:
                            platform_raw = platform_var
                        else:
                            platform_raw = platform_var[0] if platform_var.shape[0] > 0 else platform_var
                        
                        # JULD
                        juld_var = ds.variables[juld_var_name]
                        if len(juld_var.shape) == 0:
                            juld_value = juld_var
                        else:
                            juld_value = juld_var[0] if juld_var.shape[0] > 0 else juld_var
                        
                        # Coordinates
                        lat_var = ds.variables[lat_var_name]
                        lon_var = ds.variables[lon_var_name]
                        
                        if len(lat_var.shape) == 0:
                            latitude = float(lat_var)
                        else:
                            latitude = float(lat_var[0] if lat_var.shape[0] > 0 else lat_var)
                            
                        if len(lon_var.shape) == 0:
                            longitude = float(lon_var)
                        else:
                            longitude = float(lon_var[0] if lon_var.shape[0] > 0 else lon_var)
                        
                        # Load measurement arrays (these should be 1D for single profiles)
                        pressure = ds.variables[pressure_var][:]
                        temperature = ds.variables[temp_var][:] if temp_var else np.ma.masked_all_like(pressure)
                        salinity = ds.variables[salinity_var][:] if salinity_var else np.ma.masked_all_like(pressure)
                    
                    # Extract platform ID (handle both scalar and array cases)
                    if hasattr(platform_raw, 'data'):
                        if hasattr(platform_raw.data, 'shape') and platform_raw.data.shape == ():
                            # Scalar case
                            platform_id = str(platform_raw.data).strip()
                        else:
                            # Array case - decode bytes
                            if hasattr(platform_raw.data, '__iter__'):
                                try:
                                    platform_id = ''.join([
                                        b.decode('utf-8') if isinstance(b, bytes) else str(b) 
                                        for b in platform_raw.data 
                                        if b != b' ' and b != ' '
                                    ]).strip()
                                except:
                                    platform_id = str(platform_raw.data).strip()
                            else:
                                platform_id = str(platform_raw.data).strip()
                    else:
                        platform_id = str(platform_raw).strip()
                    
                    # Fallback: extract platform ID from filename if variable extraction fails
                    if not platform_id or platform_id == 'None':
                        # Extract from filename pattern like "nodc_D6902758_029.nc"
                        filename = os.path.basename(file_path)
                        platform_match = re.search(r'([0-9]{7,8})', filename)
                        if platform_match:
                            platform_id = platform_match.group(1)
                        else:
                            platform_id = filename.replace('.nc', '').replace('nodc_', '').replace('D', '').replace('R', '')
                    
                    # Reference date
                    ref_date_raw = ds.variables[ref_date_var_name][:]
                    if hasattr(ref_date_raw, 'data'):
                        if hasattr(ref_date_raw.data, '__iter__'):
                            try:
                                ref_date_str = ''.join([
                                    b.decode('utf-8') if isinstance(b, bytes) else str(b) 
                                    for b in ref_date_raw.data
                                ]).strip()
                            except:
                                ref_date_str = str(ref_date_raw.data).strip()
                        else:
                            ref_date_str = str(ref_date_raw.data).strip()
                    else:
                        ref_date_str = str(ref_date_raw).strip()
                    
                    base_date = pd.to_datetime(ref_date_str, format='%Y%m%d%H%M%S')
                    
                    # Measurement date
                    if np.ma.is_masked(juld_value) or safe_isnan(juld_value):
                        print(f"  Profile {i}: Invalid JULD, skipping")
                        continue
                    
                    measurement_date = base_date + pd.to_timedelta(float(juld_value), unit='D')
                    
                    # Coordinates validation
                    if np.isnan(latitude) or np.isnan(longitude):
                        print(f"  Profile {i}: Invalid coordinates ({latitude}, {longitude}), skipping")
                        continue
                    
                    # Check if we have ANY valid pressure data
                    try:
                        valid_pressure_mask = ~pressure.mask if hasattr(pressure, 'mask') else ~np.isnan(pressure)
                        valid_count = np.sum(valid_pressure_mask)
                    except:
                        valid_count = 0
                    
                    print(f"  Profile {i}: {valid_count} valid pressure measurements out of {len(pressure)}")
                    
                    if valid_count == 0:
                        print(f"  Profile {i}: No valid measurements, skipping")
                        continue
                    
                    # Extract valid measurements
                    profile_measurements = 0
                    for j in range(len(pressure)):
                        try:
                            # ✅ FIXED: Use safe checking to prevent ambiguous truth value errors
                            if mask_check(pressure, j) or safe_isnan(pressure[j]):
                                continue
                            
                            pressure_val = safe_float_conversion(pressure[j])
                            if pressure_val < 0:  # Pressure should be positive
                                continue
                            
                            # Temperature (can be None) - use safe checking
                            if temp_var and not (mask_check(temperature, j) or safe_isnan(temperature[j])):
                                temp_val = safe_float_conversion(temperature[j])
                            else:
                                temp_val = None
                            
                            # Salinity (can be None) - use safe checking
                            if salinity_var and not (mask_check(salinity, j) or safe_isnan(salinity[j])):
                                sal_val = safe_float_conversion(salinity[j])
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
                            
                        except Exception as e:
                            # Skip this measurement if any conversion fails
                            continue
                    
                    print(f"  Profile {i}: Extracted {profile_measurements} measurements from platform {platform_id}")
                    total_measurements += profile_measurements
                    
                except Exception as e:
                    print(f"  Profile {i}: Error - {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            print(f"Total measurements extracted: {total_measurements}")
            
            if total_measurements == 0:
                print(f"⚠️  {os.path.basename(file_path)} contains no valid measurements")
            
        return all_profiles_data
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    """Main ingestion with comprehensive error reporting"""
    pg_conn = None
    try:
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
