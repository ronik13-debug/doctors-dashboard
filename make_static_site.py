import pandas as pd
import json
import datetime
import plotly.utils
import numpy as np

# --- CONFIGURATION ---
CSV_FILE = "israel_doctors_full.csv" # Make sure this matches your filename
ISRAEL_POPULATION = 10_170_000
RETIREMENT_AGE_EXPERIENCE = 45
CURRENT_YEAR = datetime.datetime.now().year

# --- USA BENCHMARKS (AAMC 2023) ---
AAMC_USA_BENCHMARKS = {
    'רפואה פנימית': 0.376, 'רפואת המשפחה': 0.368, 'רפואת ילדים': 0.185,
    'רפואה דחופה': 0.151, 'יילוד וגינקולוגיה': 0.130, 'הרדמה': 0.128,
    'פסיכיאטריה': 0.119, 'רדיולוגיה אבחנתית': 0.085, 'כירורגיה כללית': 0.079,
    'קרדיולוגיה': 0.068, 'אנטומיה פתולוגית': 0.064, 'מחלות עיניים': 0.059,
    'כירורגיה אורתופדית': 0.058, 'אונקולוגיה': 0.054, 'גסטרואנטרולוגיה': 0.050,
    'טיפול נמרץ כללי': 0.049, 'נוירולוגיה': 0.044, 'דרמטולוגיה-מחלות עור ומין': 0.040,
    'נפרולוגיה': 0.037, 'פסיכיאטריה של הילד והמתבגר': 0.032, 'כירורגיה אורולוגית': 0.032,
    'מחלות זיהומיות': 0.032, 'מחלות אף אוזן וגרון': 0.030, 'רפואה פיזיקלית ושיקום': 0.030,
    'אנדוקרינולוגיה': 0.027, 'כירורגיה פלסטית ואסתטית': 0.023, 'רפואה לשיכוך כאב': 0.021,
    'ראומטולוגיה': 0.020, 'בריאות הציבור': 0.020, 'ניאונטולוגיה': 0.019,
    'גריאטריה': 0.019, 'נוירוכירורגיה': 0.019, 'קרדיולוגיה התערבותית': 0.017,
    'אונקולוגיה מסלול רדיותרפיה': 0.017, 'אלרגולוגיה ואימונולוגיה קלינית': 0.016,
    'כירורגיה חזה ולב': 0.014, 'מחלות ריאה': 0.014, 'כירורגית כלי דם': 0.013,
    'רפואת ספורט': 0.012,
}

def load_and_clean_data():
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"❌ Error: {CSV_FILE} not found.")
        return None

    # --- 1. NORMALIZE SPECIALTY NAMES ---
    # Define Column Names from the new CSV format
    col_s1_name = 'Specialty 1 Name'
    col_s2_name = 'Specialty 2 Name'
    
    ent_target = 'מחלות אף אוזן וגרון'
    ent_source = 'מחלות א.א.ג. וכירורגיית ראש-צוואר'
    df.loc[df[col_s1_name] == ent_source, col_s1_name] = ent_target
    df.loc[df[col_s2_name] == ent_source, col_s2_name] = ent_target

    thoracic_target = 'כירורגיה חזה ולב'
    pattern = 'חזה|לב'
    mask1 = df[col_s1_name].astype(str).str.contains(pattern, regex=True, na=False)
    df.loc[mask1, col_s1_name] = thoracic_target
    mask2 = df[col_s2_name].astype(str).str.contains(pattern, regex=True, na=False)
    df.loc[mask2, col_s2_name] = thoracic_target

    normalization_map = {
        'רפואת משפחה': 'רפואת המשפחה', 'אורתופדיה': 'כירורגיה אורתופדית',
        'עיניים': 'מחלות עיניים', 'רפואת עיניים': 'מחלות עיניים',
        'אורולוגיה': 'כירורגיה אורולוגית', 'עור ומין': 'דרמטולוגיה-מחלות עור ומין',
        'כירורגיה פלסטית': 'כירורגיה פלסטית ואסתטית', 'טיפול נמרץ': 'טיפול נמרץ כללי'
    }
    df[col_s1_name] = df[col_s1_name].replace(normalization_map)
    df[col_s2_name] = df[col_s2_name].replace(normalization_map)

    # --- 2. PARSE DATES (The Logic Change) ---
    
    # A. General Registration Date (Column B) -> Used for Retirement/Age
    df['gen_date'] = pd.to_datetime(df['Registration Date'], format='%d/%m/%Y', errors='coerce')
    
    # B. Specialty 1 Date (Column D)
    df['s1_date'] = pd.to_datetime(df['Specialty 1 Date'], format='%d/%m/%Y', errors='coerce')
    
    # C. Specialty 2 Date (Column F)
    df['s2_date'] = pd.to_datetime(df['Specialty 2 Date'], format='%d/%m/%Y', errors='coerce')

    # Drop rows where we don't even know the general registration date (corrupt data)
    df = df.dropna(subset=['gen_date'])

    # --- 3. CALCULATE GENERAL EXPERIENCE (For Retirement) ---
    df['gen_year'] = df['gen_date'].dt.year
    df['gen_experience'] = CURRENT_YEAR - df['gen_year']
    df['retirement_year'] = df['gen_year'] + RETIREMENT_AGE_EXPERIENCE
    
    return df

def generate_static_site():
    df = load_and_clean_data()
    if df is None: return

    col_s1_name = 'Specialty 1 Name'
    col_s2_name = 'Specialty 2 Name'

    # Filter Active Doctors based on GENERAL EXPERIENCE (Column B)
    active_df = df[df['gen_experience'] <= RETIREMENT_AGE_EXPERIENCE].copy()
    
    s1 = df[col_s1_name].dropna().astype(str).unique().tolist()
    s2 = df[col_s2_name].dropna().astype(str).unique().tolist()
    unique_specialties = sorted(list(set(s1 + s2)))
    unique_specialties = [s for s in unique_specialties if s.lower() not in ['nan', 'none', '']]

    dashboard_data = {}
    global_velocity_data = [] 

    print("⏳ Processing specialties...")
    
    for spec in unique_specialties:
        # --- 1. FILTERING FOR THIS SPECIALTY ---
        # Get all doctors with this specialty (History)
        mask_s1_all = df[col_s1_name] == spec
        mask_s2_all = df[col_s2_name] == spec
        spec_df_all = df[mask_s1_all | mask_s2_all].copy()
        
        # Get active doctors with this specialty (Snapshot)
        mask_s1_active = active_df[col_s1_name] == spec
        mask_s2_active = active_df[col_s2_name] == spec
        spec_df_active = active_df[mask_s1_active | mask_s2_active].copy()
        
        total_active = len(spec_df_active)
        if total_active < 30: continue 

        # --- 2. SPECIALTY JOIN DATES (The Fix) ---
        # We need to construct a list of years ONLY from the relevant specialty column
        # If doctor has Spec A in Col C, take date from Col D.
        # If doctor has Spec A in Col E, take date from Col F.
        
        dates_from_s1 = df.loc[mask_s1_all, 's1_date']
        dates_from_s2 = df.loc[mask_s2_all, 's2_date']
        
        # Combine them into one series of dates specific to THIS specialty
        all_spec_dates = pd.concat([dates_from_s1, dates_from_s2])
        spec_years = all_spec_dates.dt.year

        # --- 3. REPLACEMENT RATIO ---
        # We use general experience (Col B) for determining "Junior" vs "Veteran" status
        # because a "Junior" specialist might be a "Senior" doctor overall, but usually
        # for workforce planning, we care about age/career stage.
        juniors_count = len(spec_df_active[spec_df_active['gen_experience'] <= 10])
        veterans_count = len(spec_df_active[spec_df_active['gen_experience'] >= 30])
        
        if veterans_count > 0:
            replacement_ratio = round(juniors_count / veterans_count, 2)
        else:
            replacement_ratio = 99.9
        
        ratio_color = "#f1c40f"
        if replacement_ratio > 1.2: ratio_color = "#2ecc71"
        if replacement_ratio < 0.8: ratio_color = "#e74c3c"

        # --- 4. HEADLINE STATS ---
        velocity = (juniors_count / total_active) * 100
        outflow_now = len(spec_df_active[spec_df_active['gen_experience'] >= (RETIREMENT_AGE_EXPERIENCE - 10)])
        net_now = juniors_count - outflow_now
        
        density = (total_active / ISRAEL_POPULATION) * 1000
        usa_bench = AAMC_USA_BENCHMARKS.get(spec, None)
        usa_text = "No Benchmark"
        usa_color = "gray"
        
        if usa_bench:
            gap = density - usa_bench
            gap_docs = int(gap * (ISRAEL_POPULATION / 1000))
            if gap < 0:
                usa_text = f"Deficit: {gap_docs}"
                usa_color = "#e74c3c"
            else:
                usa_text = f"Surplus: +{gap_docs}"
                usa_color = "#2ecc71"

        global_velocity_data.append({
            'x': total_active,
            'y': velocity,
            'name': spec,
            'color': usa_color
        })

        # --- 5. FORECAST LOGIC (Using Specialty Dates for Inflow) ---
        joins_per_year = spec_years.value_counts().sort_index()
        years_idx = list(range(1980, CURRENT_YEAR + 1))
        joins_counts = [int(joins_per_year.get(y, 0)) for y in years_idx]

        # Projected Inflow (Avg of last 5 years of SPECIALTY joins)
        recent_years = range(CURRENT_YEAR - 5, CURRENT_YEAR)
        recent_inflow_sum = sum([joins_per_year.get(y, 0) for y in recent_years])
        avg_inflow = max(1, int(recent_inflow_sum / 5))

        history_years = list(range(1980, CURRENT_YEAR + 1))
        future_years = list(range(CURRENT_YEAR + 1, 2036))
        
        net_trend_history = []
        net_trend_forecast = []

        # HISTORY (1980-Current)
        for y in history_years:
            # Inflow: Use Specialty Specific Join Date
            inflow_y = len(all_spec_dates[(all_spec_dates.dt.year > (y - 10)) & (all_spec_dates.dt.year <= y)])
            
            # Outflow: Use General Retirement Date (Column B based)
            outflow_y = len(spec_df_all[(spec_df_all['retirement_year'] >= y) & (spec_df_all['retirement_year'] < (y + 10))])
            net_trend_history.append(inflow_y - outflow_y)

        # FORECAST
        for y in future_years:
            real_part_years = range(y - 10, CURRENT_YEAR + 1)
            projected_part_years = range(max(y - 10, CURRENT_YEAR + 1), y + 1)
            
            # Count real joins in the 10-year window
            count_real = len(all_spec_dates[all_spec_dates.dt.year.isin(real_part_years)])
            count_proj = len(projected_part_years) * avg_inflow
            
            inflow_forecast = count_real + count_proj
            
            # Outflow is always based on General Age
            outflow_forecast = len(spec_df_all[(spec_df_all['retirement_year'] >= y) & (spec_df_all['retirement_year'] < (y + 10))])
            
            net_trend_forecast.append(inflow_forecast - outflow_forecast)

        # EXPERIENCE STRUCTURE (General Experience)
        bins = [0, 10, 20, 30, 40]
        labels = ['Juniors (0-10)', 'Mid (10-20)', 'Senior (20-30)', 'Vet (30+)']
        exp_groups = pd.cut(spec_df_active['gen_experience'], bins=bins, labels=labels, right=False)
        exp_counts = exp_groups.value_counts().sort_index().tolist()
        
        # DENSITY
        density_x = [density]
        density_y = ['Israel']
        density_colors = ['#3498db']
        if usa_bench:
            density_x.append(usa_bench)
            density_y.append('USA')
            density_colors.append('#2c3e50')

        dashboard_data[spec] = {
            "total": int(total_active),
            "net_now": int(net_now),
            "usa_text": usa_text,
            "usa_color": usa_color,
            "ratio_val": replacement_ratio,
            "ratio_color": ratio_color,
            "charts": {
                "years_x": years_idx, 
                "years_y": joins_counts,
                "exp_x": labels,
                "exp_y": exp_counts,
                "hist_x": history_years,
                "hist_y": net_trend_history,
                "fut_x": future_years,
                "fut_y": net_trend_forecast,
                "dens_x": density_x,
                "dens_y": density_y,
                "dens_c": density_colors,
                "usa_bench": usa_bench 
            }
        }

    json_dashboard = json.dumps(dashboard_data, default=lambda x: int(x) if isinstance(x, (np.int64, np.int32)) else x)
    json_global = json.dumps(global_velocity_data, default=lambda x: int(x) if isinstance(x, (np.int64, np.int32)) else x)

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Israel Medical Workforce Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background-color: #f4f6f8; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }}
        h1 {{ text-align: center; color: #2c3e50; margin-bottom: 5px; }}
        .subtitle {{ text-align: center; color: #7f8c8d; margin-bottom: 30px; font-size: 0.9em; }}
        .section-title {{ font-size: 1.2em; font-weight: bold; color: #34495e; margin: 30px 0 15px 0; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
        .controls {{ text-align: center; margin-bottom: 30px; padding: 20px; background: #ecf0f1; border-radius: 8px; }}
        select {{ padding: 10px 20px; font-size: 16px; border-radius: 5px; border: 1px solid #bdc3c7; min-width: 300px; }}
        .kpi-row {{ display: flex; justify-content: space-around; margin-bottom: 30px; flex-wrap: wrap; gap: 10px; }}
        .kpi-card {{ background: #fff; padding: 15px 25px; border-radius: 8px; border: 1px solid #eee; text-align: center; min-width: 200px; box-shadow: 0 2px 5px rgba(0,0,0,0.03); }}
        .kpi-val {{ font-size: 24px; font-weight: bold; color: #2c3e50; display: block; }}
        .kpi-label {{ font-size: 14px; color: #95a5a6; text-transform: uppercase; letter-spacing: 1px; }}
        .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350
