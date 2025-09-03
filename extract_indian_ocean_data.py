import os
import csv
import netCDF4 as nc
import numpy as np
import pandas as pd
import re
from datetime import datetime


# --- Configuration ---
DATA_DIR = 'test_data/'
OUTPUT_CSV = 'indian_ocean_argo_data_2020_2025.csv'


def find_variable_case_insensitive(target_vars, available_vars):
    """Find variable names case-insensitively"""
    for target in target_vars:
        for available in available_vars:
            if available.lower() == target.lower():
                return available
    return None


def mask_check(array, idx):
    """Safely check if an array element is masked"""
    if hasattr(array, 'mask'):
        try:
            mask_element = array.mask[idx]
            if isinstance(mask_element, (np.ndarray, np.ma.MaskedArray)):
                return mask_element.any()
            else:
                return bool(mask_element)
        except:
            return False
    return False


def safe_isnan(value):
    """Safely check if a value contains NaN"""
    try:
        if isinstance(value, (np.ndarray, np.ma.MaskedArray)):
            if isinstance(value, np.ma.MaskedArray):
                return np.all(np.logical_or(value.mask, np.isnan(value.data)))
            else:
                return np.all(np.isnan(value))
        else:
            return np.isnan(value)
    except:
        return False


def safe_float_conversion(value):
    """Safely convert a value to float"""
    try:
        if isinstance(value, (np.ndarray, np.ma.MaskedArray)):
            if value.size == 1:
                return float(value.flat[0])
            else:
                if isinstance(value, np.ma.MaskedArray):
                    valid_indices = ~value.mask
                    if np.any(valid_indices):
                        return float(value.data[valid_indices][0])
                else:
                    valid_indices = ~np.isnan(value)
                    if np.any(valid_indices):
                        return float(value[valid_indices][0])
                return None
        else:
            return float(value)
    except:
        return None


def is_indian_ocean(latitude, longitude):
    """
    Check if coordinates are in the Indian Ocean.
    Indian Ocean bounds: 20°E to 140°E longitude, 30°N to 70°S latitude
    """
    lon_min, lon_max = 20.0, 140.0    # 20°E to 140°E
    lat_min, lat_max = -70.0, 30.0    # 70°S to 30°N
    return (lon_min <= longitude <= lon_max and lat_min <= latitude <= lat_max)


def is_year_range_2020_2025(measurement_date):
    """Check if measurement date is in 2020-2025 range"""
    try:
        year = measurement_date.year
        return 2020 <= year <= 2025
    except:
        return False


def extract_indian_ocean_data_from_file(file_path):
    """
    Extract Indian Ocean data from a single NetCDF file (2020-2025 only)
    Returns list of measurement rows
    """
    all_rows = []
    
    try:
        with nc.Dataset(file_path, 'r') as ds:
            # Determine file type
            if 'N_PROF' in ds.dimensions:
                num_profiles = len(ds.dimensions['N_PROF'])
                is_multi_profile = True
                print(f"  Multi-profile file: {num_profiles} profiles")
            else:
                num_profiles = 1
                is_multi_profile = False
                print(f"  Single-profile file")
            
            # Find variables
            available_vars = list(ds.variables.keys())
            pressure_var = find_variable_case_insensitive(['PRES_ADJUSTED', 'PRES'], available_vars)
            temp_var = find_variable_case_insensitive(['TEMP_ADJUSTED', 'TEMP'], available_vars)
            salinity_var = find_variable_case_insensitive(['PSAL_ADJUSTED', 'PSAL'], available_vars)
            platform_var_name = find_variable_case_insensitive(['PLATFORM_NUMBER', 'FLOAT_SERIAL_NO', 'WMO_INST_TYPE'], available_vars)
            juld_var_name = find_variable_case_insensitive(['JULD'], available_vars)
            lat_var_name = find_variable_case_insensitive(['LATITUDE'], available_vars)
            lon_var_name = find_variable_case_insensitive(['LONGITUDE'], available_vars)
            ref_date_var_name = find_variable_case_insensitive(['REFERENCE_DATE_TIME'], available_vars)
            
            # Check required variables
            if not all([pressure_var, platform_var_name, juld_var_name, lat_var_name, lon_var_name, ref_date_var_name]):
                print(f"  ❌ Missing required variables")
                return []
            
            indian_ocean_profiles = 0
            year_2020_2025_profiles = 0
            
            for i in range(num_profiles):
                try:
                    # Extract profile metadata
                    if is_multi_profile:
                        platform_raw = ds.variables[platform_var_name][i]
                        juld_value = ds.variables[juld_var_name][i]
                        latitude = float(ds.variables[lat_var_name][i])
                        longitude = float(ds.variables[lon_var_name][i])
                        pressure = ds.variables[pressure_var][i, :]
                        temperature = ds.variables[temp_var][i, :] if temp_var else np.ma.masked_all_like(pressure)
                        salinity = ds.variables[salinity_var][i, :] if salinity_var else np.ma.masked_all_like(pressure)
                    else:
                        platform_var = ds.variables[platform_var_name]
                        platform_raw = platform_var[0] if len(platform_var.shape) > 0 else platform_var
                        juld_var = ds.variables[juld_var_name]
                        juld_value = juld_var[0] if len(juld_var.shape) > 0 else juld_var
                        lat_var = ds.variables[lat_var_name]
                        latitude = float(lat_var[0] if len(lat_var.shape) > 0 else lat_var)
                        lon_var = ds.variables[lon_var_name]
                        longitude = float(lon_var[0] if len(lon_var.shape) > 0 else lon_var)
                        pressure = ds.variables[pressure_var][:]
                        temperature = ds.variables[temp_var][:] if temp_var else np.ma.masked_all_like(pressure)
                        salinity = ds.variables[salinity_var][:] if salinity_var else np.ma.masked_all_like(pressure)
                    
                    # Extract platform ID
                    if hasattr(platform_raw, 'data'):
                        if hasattr(platform_raw.data, 'shape') and platform_raw.data.shape == ():
                            platform_id = str(platform_raw.data).strip()
                        else:
                            if hasattr(platform_raw.data, '__iter__'):
                                platform_id = ''.join([b.decode('utf-8') if isinstance(b, bytes) else str(b) for b in platform_raw.data if b != b' ' and b != ' ']).strip()
                            else:
                                platform_id = str(platform_raw.data).strip()
                    else:
                        platform_id = str(platform_raw).strip()
                    
                    # Fallback platform ID from filename
                    if not platform_id or platform_id == 'None':
                        filename = os.path.basename(file_path)
                        platform_match = re.search(r'([0-9]{7,8})', filename)
                        platform_id = platform_match.group(1) if platform_match else filename.replace('.nc', '')
                    
                    # Extract date
                    ref_date_raw = ds.variables[ref_date_var_name][:]
                    if hasattr(ref_date_raw, 'data'):
                        if hasattr(ref_date_raw.data, '__iter__'):
                            ref_date_str = ''.join([b.decode('utf-8') if isinstance(b, bytes) else str(b) for b in ref_date_raw.data]).strip()
                        else:
                            ref_date_str = str(ref_date_raw.data).strip()
                    else:
                        ref_date_str = str(ref_date_raw).strip()
                    
                    base_date = pd.to_datetime(ref_date_str, format='%Y%m%d%H%M%S')
                    
                    if np.ma.is_masked(juld_value) or safe_isnan(juld_value):
                        continue
                    
                    measurement_date = base_date + pd.to_timedelta(float(juld_value), unit='D')
                    
                    # **FILTER 1: Check year range (2020-2025)**
                    if not is_year_range_2020_2025(measurement_date):
                        continue
                    year_2020_2025_profiles += 1
                    
                    # **FILTER 2: Check Indian Ocean coordinates**
                    if not is_indian_ocean(latitude, longitude):
                        continue
                    indian_ocean_profiles += 1
                    
                    # Extract measurements
                    profile_measurements = 0
                    for j in range(len(pressure)):
                        try:
                            if mask_check(pressure, j) or safe_isnan(pressure[j]):
                                continue
                            
                            pressure_val = safe_float_conversion(pressure[j])
                            if pressure_val is None or pressure_val < 0:
                                continue
                            
                            # Temperature
                            if temp_var and not (mask_check(temperature, j) or safe_isnan(temperature[j])):
                                temp_val = safe_float_conversion(temperature[j])
                            else:
                                temp_val = None
                            
                            # Salinity
                            if salinity_var and not (mask_check(salinity, j) or safe_isnan(salinity[j])):
                                sal_val = safe_float_conversion(salinity[j])
                            else:
                                sal_val = None
                            
                            # Add row to results
                            all_rows.append([
                                platform_id,
                                measurement_date.strftime('%Y-%m-%d %H:%M:%S'),
                                latitude,
                                longitude,
                                pressure_val,
                                temp_val,
                                sal_val,
                                measurement_date.year  # Add year column for reference
                            ])
                            profile_measurements += 1
                            
                        except Exception as e:
                            continue
                    
                    if profile_measurements > 0:
                        print(f"    Profile {i}: {profile_measurements} measurements from {measurement_date.year}")
                    
                except Exception as e:
                    continue
            
            print(f"  📊 Summary: {year_2020_2025_profiles} profiles in 2020-2025, {indian_ocean_profiles} in Indian Ocean")
        
        return all_rows
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return []


def main():
    """
    Main extraction function: processes all NetCDF files and saves to CSV
    """
    print("🌊 INDIAN OCEAN ARGO DATA EXTRACTION (2020-2025)")
    print("=" * 60)
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ Directory '{DATA_DIR}' not found")
        return
    
    nc_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.nc')]
    if not nc_files:
        print(f"❌ No .nc files in '{DATA_DIR}'")
        return
    
    print(f"Found {len(nc_files)} NetCDF files to process")
    print(f"Filtering for: Indian Ocean (20°E-140°E, 70°S-30°N) AND years 2020-2025")
    
    all_extracted_rows = []
    files_with_data = 0
    total_files_processed = 0
    
    for file_path in nc_files:
        print(f"\n📂 Processing: {os.path.basename(file_path)}")
        
        file_rows = extract_indian_ocean_data_from_file(file_path)
        
        if file_rows:
            all_extracted_rows.extend(file_rows)
            files_with_data += 1
            print(f"  ✅ Extracted {len(file_rows)} measurements")
        else:
            print(f"  ❌ No valid Indian Ocean 2020-2025 data found")
        
        total_files_processed += 1
    
    # Write to CSV
    if all_extracted_rows:
        print(f"\n💾 Writing {len(all_extracted_rows):,} measurements to {OUTPUT_CSV}...")
        
        # CSV headers
        headers = [
            'platform_id', 
            'measurement_date', 
            'latitude', 
            'longitude', 
            'pressure_dbar', 
            'temperature_celsius', 
            'salinity_psu',
            'year'
        ]
        
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)  # Write header
            writer.writerows(all_extracted_rows)  # Write data
        
        print(f"✅ Data extraction complete!")
        print(f"\n📊 EXTRACTION SUMMARY:")
        print(f"   Files processed: {total_files_processed}")
        print(f"   Files with data: {files_with_data}")
        print(f"   Total measurements: {len(all_extracted_rows):,}")
        print(f"   Output file: {OUTPUT_CSV}")
        print(f"   File size: {os.path.getsize(OUTPUT_CSV) / (1024*1024):.2f} MB")
        
        # Show year distribution
        year_counts = {}
        for row in all_extracted_rows:
            year = row[7]  # year column
            year_counts[year] = year_counts.get(year, 0) + 1
        
        print(f"\n📅 Year Distribution:")
        for year in sorted(year_counts.keys()):
            print(f"   {year}: {year_counts[year]:,} measurements")
    
    else:
        print(f"\n⚠️  No Indian Ocean data found in any files for 2020-2025 period")


if __name__ == "__main__":
    main()
