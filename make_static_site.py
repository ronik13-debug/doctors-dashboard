import pandas as pd
import json
import datetime
import plotly.utils
import numpy as np
import requests
import os

# --- CONFIGURATION ---
API_RESOURCE_ID = "9c64c522-bbc2-48fe-96fb-3b2a8626f59e"
ISRAEL_POPULATION = 10_170_000
RETIREMENT_AGE_EXPERIENCE = 45
CURRENT_YEAR = datetime.datetime.now().year
TIMESTAMP = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

# --- USA BENCHMARKS (Doctors per 1,000) ---
AAMC_USA_BENCHMARKS = {
    'רפואה פנימית': 0.807,      
    'רפואת המשפחה': 0.316,      
    'רפואת ילדים': 0.357,       
    'רפואה דחופה': 0.143,       
    'יילוד וגינקולוגיה': 0.157, 
    'הרדמה': 0.192,             
    'פסיכיאטריה': 0.174,        
    'נוירולוגיה': 0.075,        
    'רדיולוגיה אבחנתית': 0.203, 
    'כירורגיה כללית': 0.136,    
    'אנטומיה פתולוגית': 0.102,  
    'מחלות עיניים': 0.077,      
    'כירורגיה אורתופדית': 0.095,
    'כירורגיה פלסטית ואסתטית': 0.027, 
    'כירורגיה אורולוגית': 0.036,
    'מחלות א.א.ג. וכירורגיית ראש-צוואר': 0.048, 
    'רפואה פיזיקלית ושיקום': 0.039, 
    'דרמטולוגיה-מחלות עור ומין': 0.048, 
    'בריאות הציבור': 0.048,     
    'נוירוכירורגיה': 0.022,     
    'כירורגיה חזה ולב': 0.018,  
    'אלרגולוגיה ואימונולוגיה קלינית': 0.019
}

# --- US NEW LICENSES DATA (Source: ABMS Report 2024-2025) ---
US_NEW_LICENSES = {
    'Allergy and Immunology': {2015: 147, 2016: 134, 2017: 148, 2018: 145, 2019: 142, 2020: 136, 2021: 143, 2022: 164, 2023: 165, 2024: 158},
    'Anesthesiology': {2015: 1570, 2016: 1654, 2017: 1620, 2018: 1600, 2019: 1754, 2020: 257, 2021: 2724, 2022: 1824, 2023: 1689, 2024: 1782},
    'Colon and Rectal Surgery': {2015: 80, 2016: 99, 2017: 86, 2018: 90, 2019: 97, 2020: 0, 2021: 201, 2022: 107, 2023: 120, 2024: 108},
    'Dermatology': {2015: 432, 2016: 434, 2017: 498, 2018: 523, 2019: 540, 2020: 533, 2021: 544, 2022: 555, 2023: 553, 2024: 540},
    'Emergency Medicine': {2015: 1682, 2016: 1778, 2017: 1868, 2018: 1943, 2019: 2129, 2020: 0, 2021: 4406, 2022: 2310, 2023: 2748, 2024: 2602},
    'Family Medicine': {2015: 3506, 2016: 3599, 2017: 3567, 2018: 3885, 2019: 3805, 2020: 3898, 2021: 4259, 2022: 4339, 2023: 4456, 2024: 4450},
    'Internal Medicine': {2015: 7776, 2016: 7857, 2017: 8154, 2018: 8773, 2019: 9003, 2020: 9058, 2021: 8854, 2022: 9461, 2023: 9750, 2024: 10241},
    'Neurological Surgery': {2015: 151, 2016: 149, 2017: 145, 2018: 224, 2019: 214, 2020: 186, 2021: 246, 2022: 211, 2023: 183, 2024: 178},
    'Nuclear Medicine': {2015: 63, 2016: 43, 2017: 49, 2018: 44, 2019: 62, 2020: 53, 2021: 53, 2022: 60, 2023: 60, 2024: 50},
    'Obstetrics and Gynecology': {2015: 1261, 2016: 1277, 2017: 1296, 2018: 1105, 2019: 1342, 2020: 595, 2021: 2017, 2022: 1845, 2023: 1351, 2024: 1277},
    'Ophthalmology': {2015: 451, 2016: 442, 2017: 501, 2018: 571, 2019: 528, 2020: 470, 2021: 576, 2022: 494, 2023: 501, 2024: 500},
    'Orthopaedic Surgery': {2015: 707, 2016: 700, 2017: 689, 2018: 697, 2019: 755, 2020: 621, 2021: 751, 2022: 708, 2023: 777, 2024: 769},
    'Otolaryngology - Head and Neck Surgery': {2015: 309, 2016: 289, 2017: 292, 2018: 293, 2019: 313, 2020: 0, 2021: 638, 2022: 331, 2023: 333, 2024: 0},
    'Pathology': {2015: 660, 2016: 606, 2017: 632, 2018: 619, 2019: 559, 2020: 501, 2021: 708, 2022: 657, 2023: 609, 2024: 558},
    'Pediatrics': {2015: 3180, 2016: 2891, 2017: 3498, 2018: 3544, 2019: 3212, 2020: 3130, 2021: 3129, 2022: 3166, 2023: 3326, 2024: 3712},
    'Physical Medicine and Rehabilitation': {2015: 348, 2016: 376, 2017: 451, 2018: 411, 2019: 459, 2020: 354, 2021: 422, 2022: 395, 2023: 448, 2024: 439},
    'Plastic Surgery': {2015: 203, 2016: 195, 2017: 213, 2018: 209, 2019: 167, 2020: 191, 2021: 183, 2022: 235, 2023: 206, 2024: 206},
    'Preventive Medicine': {2015: 198, 2016: 198, 2017: 205, 2018: 177, 2019: 216, 2020: 218, 2021: 222, 2022: 205, 2023: 183, 2024: 174},
    'Psychiatry': {2015: 1490, 2016: 1465, 2017: 1447, 2018: 1478, 2019: 1595, 2020: 1504, 2021: 1844, 2022: 1869, 2023: 2009, 2024: 2024},
    'Neurology': {2015: 689, 2016: 644, 2017: 688, 2018: 711, 2019: 743, 2020: 790, 2021: 796, 2022: 804, 2023: 901, 2024: 967},
    'Diagnostic Radiology': {2015: 1092, 2016: 1273, 2017: 1150, 2018: 1176, 2019: 1104, 2020: 0, 2021: 2104, 2022: 1074, 2023: 1082, 2024: 1113},
    'Surgery': {2015: 1046, 2016: 1341, 2017: 1068, 2018: 910, 2019: 1492, 2020: 806, 2021: 1516, 2022: 1373, 2023: 1362, 2024: 1340},
    'Thoracic Surgery': {2015: 86, 2016: 101, 2017: 109, 2018: 102, 2019: 117, 2020: 0, 2021: 224, 2022: 133, 2023: 135, 2024: 139},
    'Urology': {2015: 240, 2016: 291, 2017: 277, 2018: 285, 2019: 305, 2020: 314, 2021: 293, 2022: 326, 2023: 331, 2024: 341},
    'Vascular Surgery': {2015: 136, 2016: 151, 2017: 139, 2018: 159, 2019: 169, 2020: 0, 2021: 341, 2022: 165, 2023: 188, 2024: 188},
    'Cardiovascular Disease': {2015: 929, 2016: 896, 2017: 934, 2018: 980, 2019: 933, 2020: 932, 2021: 988, 2022: 1035, 2023: 1074, 2024: 1112},
    'Gastroenterology': {2015: 509, 2016: 517, 2017: 533, 2018: 561, 2019: 551, 2020: 551, 2021: 616, 2022: 612, 2023: 641, 2024: 658},
    'Hematology': {2015: 397, 2016: 497, 2017: 492, 2018: 463, 2019: 512, 2020: 488, 2021: 512, 2022: 521, 2023: 561, 2024: 555},
    'Critical Care Medicine': {2015: 525, 2016: 740, 2017: 682, 2018: 745, 2019: 678, 2020: 733, 2021: 775, 2022: 818, 2023: 815, 2024: 900},
    'Pulmonary Disease': {2015: 641, 2016: 597, 2017: 628, 2018: 597, 2019: 628, 2020: 653, 2021: 705, 2022: 739, 2023: 720, 2024: 757},
    'Rheumatology': {2015: 210, 2016: 230, 2017: 204, 2018: 243, 2019: 260, 2020: 240, 2021: 245, 2022: 250, 2023: 248, 2024: 307},
    'Infectious Disease': {2015: 367, 2016: 373, 2017: 340, 2018: 344, 2019: 368, 2020: 346, 2021: 359, 2022: 382, 2023: 410, 2024: 406},
    'Endocrinology, Diabetes and Metabolism': {2015: 307, 2016: 289, 2017: 302, 2018: 312, 2019: 316, 2020: 318, 2021: 299, 2022: 322, 2023: 368, 2024: 373},
    'Medical Oncology': {2015: 582, 2016: 609, 2017: 612, 2018: 542, 2019: 568, 2020: 598, 2021: 603, 2022: 664, 2023: 685, 2024: 693},
    'Nephrology': {2015: 487, 2016: 438, 2017: 377, 2018: 353, 2019: 321, 2020: 353, 2021: 355, 2022: 388, 2023: 379, 2024: 398}
}

# --- US TOTAL ACTIVE PHYSICIANS (Denominator) ---
US_TOTAL_ACTIVE = {
    'Colon and Rectal Surgery': 2839,
    'Medical Genetics and Genomics': 2890,
    'Nuclear Medicine': 4394,
    'Thoracic Surgery': 5999,
    'Allergy and Immunology': 6501,
    'Neurological Surgery': 7468,
    'Plastic Surgery': 9119,
    'Urology': 12102,
    'Physical Medicine and Rehabilitation': 12983,
    'Dermatology': 16090,
    'Preventive Medicine': 16126,
    'Otolaryngology - Head and Neck Surgery': 16160,
    'Ophthalmology': 25888,
    'Orthopaedic Surgery': 31920,
    'Pathology': 34114,
    'Surgery': 45760,
    'Emergency Medicine': 48182,
    'Obstetrics and Gynecology': 52809,
    'Anesthesiology': 64537,
    'Radiology': 68325,
    'Psychiatry': 61000, 
    'Neurology': 22554, 
    'Family Medicine': 106055,
    'Pediatrics': 119989,
    'Internal Medicine': 271213,
    'Cardiovascular Disease': 271213,
    'Gastroenterology': 271213,
    'Hematology': 271213,
    'Critical Care Medicine': 271213,
    'Pulmonary Disease': 271213,
    'Rheumatology': 271213,
    'Infectious Disease': 271213,
    'Endocrinology, Diabetes and Metabolism': 271213,
    'Medical Oncology': 271213,
    'Nephrology': 271213,
    'Diagnostic Radiology': 68325,
    'Vascular Surgery': 45760 
}

# Mapping Israeli keys to US keys
US_MAPPING = {
    'רפואה פנימית': 'Internal Medicine',
    'רפואת המשפחה': 'Family Medicine',
    'רפואת ילדים': 'Pediatrics',
    'רפואה דחופה': 'Emergency Medicine',
    'יילוד וגינקולוגיה': 'Obstetrics and Gynecology',
    'הרדמה': 'Anesthesiology',
    'פסיכיאטריה': 'Psychiatry',
    'נוירולוגיה': 'Neurology',
    'רדיולוגיה אבחנתית': 'Diagnostic Radiology',
    'כירורגיה כללית': 'Surgery',
    'אנטומיה פתולוגית': 'Pathology',
    'מחלות עיניים': 'Ophthalmology',
    'כירורגיה אורתופדית': 'Orthopaedic Surgery',
    'כירורגיה פלסטית ואסתטית': 'Plastic Surgery',
    'כירורגיה אורולוגית': 'Urology',
    'מחלות א.א.ג. וכירורגיית ראש-צוואר': 'Otolaryngology - Head and Neck Surgery',
    'רפואה פיזיקלית ושיקום': 'Physical Medicine and Rehabilitation',
    'דרמטולוגיה-מחלות עור ומין': 'Dermatology',
    'בריאות הציבור': 'Preventive Medicine',
    'נוירוכירורגיה': 'Neurological Surgery',
    'כירורגיה חזה ולב': 'Thoracic Surgery',
    'אלרגולוגיה ואימונולוגיה קלינית': 'Allergy and Immunology',
    'קרדיולוגיה': 'Cardiovascular Disease',
    'גסטרואנטרולוגיה': 'Gastroenterology',
    'המטולוגיה': 'Hematology',
    'אונקולוגיה': 'Medical Oncology',
    'טיפול נמרץ כללי': 'Critical Care Medicine',
    'מחלות ריאה': 'Pulmonary Disease',
    'ראומטולוגיה': 'Rheumatology',
    'מחלות זיהומיות': 'Infectious Disease',
    'אנדוקרינולוגיה': 'Endocrinology, Diabetes and Metabolism',
    'נפרולוגיה': 'Nephrology',
    'כירורגית כלי דם': 'Vascular Surgery'
}

def get_year_simple(val):
    s = str(val).strip()
    if s.endswith('.0'): s = s[:-2]
    if not s or s.lower() in ['nan', 'none', '', 'nat']:
        return np.nan
    if len(s) >= 4:
        potential_year = s[-4:]
        if potential_year.isdigit():
            y = int(potential_year)
            if 1900 <= y <= CURRENT_YEAR + 1:
                return y
    return np.nan

def load_and_clean_data():
    print("⏳ Connecting to data.gov.il API...")
    api_url = "https://data.gov.il/api/3/action/datastore_search"
    limit = 32000 
    offset = 0
    all_records = []
    
    while True:
        params = {"resource_id": API_RESOURCE_ID, "limit": limit, "offset": offset}
        try:
            r = requests.get(api_url, params=params, timeout=45)
            r.raise_for_status()
            data = r.json()
            if not data.get('success'): break
            records = data['result']['records']
            if not records: break
            all_records.extend(records)
            offset += limit
            print(f"   Fetched {len(all_records)} rows...", end='\r')
            if len(records) < limit: break
        except Exception as e:
            print(f"\n❌ Error fetching API: {e}")
            return None

    df = pd.DataFrame(all_records)
    print(f"\n✅ Total Raw Records: {len(df)}")
    
    col_map = {
        'שם פרטי': 'first_name', 'שם משפחה': 'last_name',
        'מספר רישיון': 'license_num', 'מספר רשיון': 'license_num', 'mispar_rishyon': 'license_num',
        'תאריך רישום רישיון': 'license_date_raw', 'תאריך רישיון': 'license_date_raw',
        'שם התמחות': 'specialty_name', 'תאור מומחיות': 'specialty_name',
        'תאריך רישום התמחות': 'spec_date_raw'
    }
    df = df.rename(columns=col_map)
    
    if 'first_name' in df.columns:
        df['first_name'] = df['first_name'].astype(str).str.strip()
        df['last_name'] = df['last_name'].astype(str).str.strip()
        df['Name'] = df['first_name'] + " " + df['last_name']
    else:
        df['Name'] = "Unknown"

    if 'license_num' not in df.columns:
        df['license_num'] = df['Name'] + "_" + df['license_date_raw'].astype(str)

    print("⏳ Extracting years...")
    if 'license_date_raw' in df.columns:
        df['gen_year'] = df['license_date_raw'].apply(get_year_simple)
    else:
        print("❌ Critical: No license date column found.")
        return None
    
    if 'spec_date_raw' in df.columns:
        df['spec_year'] = df['spec_date_raw'].apply(get_year_simple)
    else:
        df['spec_year'] = np.nan

    df = df.dropna(subset=['gen_year'])

    if 'specialty_name' not in df.columns: df['specialty_name'] = "Unknown"
    df['specialty_name'] = df['specialty_name'].astype(str).str.strip()
    
    normalization_map = {
        'מחלות אף אוזן וגרון': 'מחלות א.א.ג. וכירורגיית ראש-צוואר',
        'כירורגית בית החזה - מסלול כירורגית לב': 'כירורגית לב',
        'כירורגיה של בית החזה - מסלול לב מבוגרים': 'כירורגית לב',
        'כירורגיה של בית החזה - מסלול לב ילדים': 'כירורגית לב ילדים',
        'כירורגית בית החזה - מסלול כירורגית לב וכירורגית חזה כללית': 'כירורגית חזה ולב',
        'כירורגית בית החזה - מסלול כירורגית חזה כללית': 'כירורגיה של בית החזה',
        'נוירולוגיית ילדים': 'נוירולוגית ילדים והתפתחות הילד',
        'רפואה דחופה - מסלול מבוגרים': 'רפואה דחופה',
        'רפואת משפחה': 'רפואת המשפחה', 
        'אורתופדיה': 'כירורגיה אורתופדית',
        'עיניים': 'מחלות עיניים', 'רפואת עיניים': 'מחלות עיניים',
        'אורולוגיה': 'כירורגיה אורולוגית', 
        'עור ומין': 'דרמטולוגיה-מחלות עור ומין',
        'כירורגיה פלסטית': 'כירורגיה פלסטית ואסתטית', 
        'טיפול נמרץ': 'טיפול נמרץ כללי'
    }
    df['specialty_name'] = df['specialty_name'].replace(normalization_map)

    df['gen_experience'] = CURRENT_YEAR - df['gen_year']
    df['spec_experience'] = CURRENT_YEAR - df['spec_year']
    df['spec_experience'] = df['spec_experience'].fillna(df['gen_experience']) 
    df['retirement_year_spec'] = df['spec_year'].fillna(df['gen_year']) + RETIREMENT_AGE_EXPERIENCE

    return df

def generate_static_site():
    df = load_and_clean_data()
    if df is None: return

    active_df_rows = df[df['gen_experience'] <= RETIREMENT_AGE_EXPERIENCE].copy()
    unique_specialties = sorted([s for s in df['specialty_name'].unique() if s.lower() not in ['nan', 'none', '', 'unknown']])
    
    dashboard_data = {}
    global_velocity_data = [] 

    print("⏳ Generating Dashboard...")
    
    for spec in unique_specialties:
        spec_df_all = df[df['specialty_name'] == spec]
        spec_df_active = active_df_rows[active_df_rows['specialty_name'] == spec]
        
        total_active = spec_df_active['license_num'].nunique()
        if total_active < 30: continue 

        unique_active_docs = spec_df_active.drop_duplicates(subset='license_num')
        spec_df_all_unique = spec_df_all.drop_duplicates(subset='license_num')

        count_over_45 = len(spec_df_all_unique[spec_df_all_unique['gen_experience'] > 45])

        juniors_count = len(unique_active_docs[unique_active_docs['gen_experience'] <= 10])
        veterans_count = len(unique_active_docs[unique_active_docs['gen_experience'] >= 30])
        replacement_ratio = round(juniors_count / veterans_count, 2) if veterans_count > 0 else 99.9
        
        ratio_color = "#f39c12"
        if replacement_ratio > 1.2: ratio_color = "#27ae60"
        if replacement_ratio < 0.8: ratio_color = "#e74c3c"

        velocity = (juniors_count / total_active) * 100
        outflow_now = len(unique_active_docs[unique_active_docs['gen_experience'] >= (RETIREMENT_AGE_EXPERIENCE - 10)])
        net_now = juniors_count - outflow_now
        
        density = (total_active / ISRAEL_POPULATION) * 1000
        usa_bench = AAMC_USA_BENCHMARKS.get(spec, None)
        usa_text, usa_color = "No Benchmark", "#95a5a6"
        if usa_bench:
            gap = density - usa_bench
            gap_docs = int(gap * (ISRAEL_POPULATION / 1000))
            if gap < 0: usa_text, usa_color = f"Deficit: {gap_docs}", "#e74c3c"
            else: usa_text, usa_color = f"Surplus: +{gap_docs}", "#27ae60"

        global_velocity_data.append({'x': total_active, 'y': velocity, 'name': spec, 'color': usa_color})

        spec_start_years = spec_df_all_unique['spec_year'].dropna().astype(int)
        joins_per_year = spec_start_years.value_counts().sort_index()
        years_idx = list(range(1980, CURRENT_YEAR + 1))
        joins_counts = [int(joins_per_year.get(y, 0)) for y in years_idx]

        us_x = []
        us_y = []
        us_name = US_MAPPING.get(spec)
        normalization_factor = 1.0

        if us_name and us_name in US_NEW_LICENSES and us_name in US_TOTAL_ACTIVE:
            us_dict = US_NEW_LICENSES[us_name]
            us_total_active = US_TOTAL_ACTIVE[us_name]
            il_total_active = total_active
            
            if us_total_active > 0:
                normalization_factor = il_total_active / us_total_active
            
            us_x = sorted(list(us_dict.keys()))
            us_y = [us_dict[y] for y in us_x]

        recent_years = range(CURRENT_YEAR - 5, CURRENT_YEAR)
        recent_inflow_sum = sum([joins_per_year.get(y, 0) for y in recent_years])
        avg_inflow = max(1, int(recent_inflow_sum / 5))

        history_years = list(range(1980, CURRENT_YEAR + 1))
        future_years = list(range(CURRENT_YEAR + 1, CURRENT_YEAR + 5))
        net_trend_history, net_trend_forecast = [], []
        spec_retire_years = spec_df_all_unique['retirement_year_spec'].dropna().astype(int)

        for y in history_years:
            inflow = len(spec_start_years[(spec_start_years > (y - 8)) & (spec_start_years <= y)])
            outflow = len(spec_retire_years[(spec_retire_years >= y) & (spec_retire_years < (y - 8))])
            net_trend_history.append(inflow - outflow)

        for y in future_years:
            real_part = range(y - 8, CURRENT_YEAR + 1)
            proj_part = range(max(y - 8, CURRENT_YEAR + 1), y + 1)
            count_real = len(spec_start_years[spec_start_years.isin(real_part)])
            count_proj = len(proj_part) * avg_inflow
            outflow = len(spec_retire_years[(spec_retire_years >= y) & (spec_retire_years < (y - 8))])
            net_trend_forecast.append((count_real + count_proj) - outflow)

        bins = [0, 10, 25, 120]
        labels = ['Juniors (0-10y)', 'Mid (10-25y)', 'Seniors (25-45y)']
        exp_groups = pd.cut(unique_active_docs['spec_experience'], bins=bins, labels=labels, right=False)
        exp_counts = exp_groups.value_counts().sort_index()
        pie_labels = exp_counts.index.tolist()
        pie_values = exp_counts.values.tolist()
        
        density_x = [density]
        density_
