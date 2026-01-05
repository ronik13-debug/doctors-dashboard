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

def parse_custom_date(date_str):
    """
    Robust parser handling:
    1. ddmmyyyy (No separators)
    2. dd/mm/yyyy (Slashes - Israeli standard)
    3. yyyy-mm-dd (ISO standard)
    """
    if pd.isna(date_str) or str(date_str).strip() == "":
        return pd.NaT
    
    s = str(date_str).strip()
    
    # Case 1: 7 or 8 digits (ddmmyyyy)
    if s.isdigit() and (len(s) == 7 or len(s) == 8):
        if len(s) == 7: s = "0" + s # Pad leading zero
        try:
            day = int(s[0:2])
            month = int(s[2:4])
            year = int(s[4:8])
            return pd.Timestamp(year=year, month=month, day=day)
        except:
            return pd.NaT
            
    # Case 2: Standard Format with separators (Force DayFirst!)
    return pd.to_datetime(s, errors='coerce', dayfirst=True)

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
    
    # Construct Name
    if 'first_name' in df.columns:
        df['first_name'] = df['first_name'].astype(str).str.strip()
        df['last_name'] = df['last_name'].astype(str).str.strip()
        df['Name'] = df['first_name'] + " " + df['last_name']
    else:
        df['Name'] = "Unknown"

    # Fallback ID
    if 'license_num' not in df.columns:
        df['license_num'] = df['Name'] + "_" + df['license_date_raw'].astype(str)

    # Parse Dates
    print("⏳ Parsing dates...")
    if 'license_date_raw' in df.columns:
        df['gen_date'] = df['license_date_raw'].apply(parse_custom_date)
    else:
        print("❌ Critical: No license date column found.")
        return None
    
    if 'spec_date_raw' in df.columns:
        df['spec_date'] = df['spec_date_raw'].apply(parse_custom_date)
    else:
        df['spec_date'] = pd.NaT

    df = df.dropna(subset=['gen_date'])

    # Normalize Specialties
    if 'specialty_name' not in df.columns: df['specialty_name'] = "Unknown"
    df['specialty_name'] = df['specialty_name'].astype(str).str.strip()
    
    ent_target = 'מחלות אף אוזן וגרון'
    df.loc[df['specialty_name'] == 'מחלות א.א.ג. וכירורגיית ראש-צוואר', 'specialty_name'] = ent_target
    df.loc[df['specialty_name'].str.contains('חזה|לב', regex=True), 'specialty_name'] = 'כירורגיה חזה ולב'

    normalization_map = {
        'רפואת משפחה': 'רפואת המשפחה', 'אורתופדיה': 'כירורגיה אורתופדית',
        'עיניים': 'מחלות עיניים', 'רפואת עיניים': 'מחלות עיניים',
        'אורולוגיה': 'כירורגיה אורולוגית', 'עור ומין': 'דרמטולוגיה-מחלות עור ומין',
        'כירורגיה פלסטית': 'כירורגיה פלסטית ואסתטית', 'טיפול נמרץ': 'טיפול נמרץ כללי'
    }
    df['specialty_name'] = df['specialty_name'].replace(normalization_map)

    # CALCULATE EXPERIENCE
    df['gen_year'] = df['gen_date'].dt.year
    df['gen_experience'] = CURRENT_YEAR - df['gen_year']
    
    # Specialty Specific Experience (Preferred)
    df['spec_year'] = df['spec_date'].dt.year
    df['spec_experience'] = CURRENT_YEAR - df['spec_year']
    
    # Fallback to General Experience only if Spec Date is missing
    df['spec_experience'] = df['spec_experience'].fillna(df['gen_experience']) 

    df['retirement_year_spec'] = df['spec_year'].fillna(df['gen_year']) + RETIREMENT_AGE_EXPERIENCE

    return df

def generate_static_site():
    df = load_and_clean_data()
    if df is None: return

    # Filter Active Doctors
    active_df_rows = df[df['gen_experience'] <= RETIREMENT_AGE_EXPERIENCE].copy()
    unique_specialties = sorted([s for s in df['specialty_name'].unique() if s.lower() not in ['nan', 'none', '', 'unknown']])
    
    dashboard_data = {}
    global_velocity_data = [] 

    print("⏳ Generating Dashboard...")
    
    for spec in unique_specialties:
        # Full history for this specialty
        spec_df_all = df[df['specialty_name'] == spec]
        # Active snapshot for this specialty
        spec_df_active = active_df_rows[active_df_rows['specialty_name'] == spec]
        
        # Count UNIQUE doctors (Total Active)
        total_active = spec_df_active['license_num'].nunique()
        if total_active < 30: continue 

        # Create unique docs list for KPIs (deduplicated)
        unique_active_docs = spec_df_active.drop_duplicates(subset='license_num')
        
        # KPIs
        juniors_count = len(unique_active_docs[unique_active_docs['gen_experience'] <= 10])
        veterans_count = len(unique_active_docs[unique_active_docs['gen_experience'] >= 30])
        replacement_ratio = round(juniors_count / veterans_count, 2) if veterans_count > 0 else 99.9
        
        ratio_color = "#f1c40f"
        if replacement_ratio > 1.2: ratio_color = "#2ecc71"
        if replacement_ratio < 0.8: ratio_color = "#e74c3c"

        velocity = (juniors_count / total_active) * 100
        outflow_now = len(unique_active_docs[unique_active_docs['gen_experience'] >= (RETIREMENT_AGE_EXPERIENCE - 10)])
        net_now = juniors_count - outflow_now
        
        density = (total_active / ISRAEL_POPULATION) * 1000
        usa_bench = AAMC_USA_BENCHMARKS.get(spec, None)
        usa_text, usa_color = "No Benchmark", "gray"
        if usa_bench:
            gap = density - usa_bench
            gap_docs = int(gap * (ISRAEL_POPULATION / 1000))
            if gap < 0: usa_text, usa_color = f"Deficit: {gap_docs}", "#e74c3c"
            else: usa_text, usa_color = f"Surplus: +{gap_docs}", "#2ecc71"

        global_velocity_data.append({'x': total_active, 'y': velocity, 'name': spec, 'color': usa_color})

        # --- PREPARE CHART DATA (With Deduplication!) ---
        # Ensure we don't count the same doctor twice if they appear in the API twice for the same spec
        spec_df_all_unique = spec_df_all.drop_duplicates(subset='license_num')
        
        # CHART 1: New Licenses (Using Specialty Date)
        spec_start_years = spec_df_all_unique['spec_year'].dropna()
        joins_per_year = spec_start_years.value_counts().sort_index()
        years_idx = list(range(1980, CURRENT_YEAR + 1))
        joins_counts = [int(joins_per_year.get(y, 0)) for y in years_idx]

        # Forecast logic
        recent_years = range(CURRENT_YEAR - 5, CURRENT_YEAR)
        recent_inflow_sum = sum([joins_per_year.get(y, 0) for y in recent_years])
        avg_inflow = max(1, int(recent_inflow_sum / 5))

        # CHART 2: Net Pipeline (Inflow vs Outflow)
        history_years = list(range(1980, CURRENT_YEAR + 1))
        future_years = list(range(CURRENT_YEAR + 1, 2036))
        net_trend_history, net_trend_forecast = [], []
        spec_retire_years = spec_df_all_unique['retirement_year_spec'].dropna()

        for y in history_years:
            inflow = len(spec_start_years[(spec_start_years > (y - 10)) & (spec_start_years <= y)])
            outflow = len(spec_retire_years[(spec_retire_years >= y) & (spec_retire_years < (y + 10))])
            net_trend_history.append(inflow - outflow)

        for y in future_years:
            real_part = range(y - 10, CURRENT_YEAR + 1)
            proj_part = range(max(y - 10, CURRENT_YEAR + 1), y + 1)
            count_real = len(spec_start_years[spec_start_years.isin(real_part)])
            count_proj = len(proj_part) * avg_inflow
            outflow = len(spec_retire_years[(spec_retire_years >= y) & (spec_retire_years < (y + 10))])
            net_trend_forecast.append((count_real + count_proj) - outflow)

        # CHART 3: Experience Structure (Pie)
        bins = [0, 10, 20, 45, 120]
        labels = ['Juniors (0-10y)', 'Mid (10-20y)', 'Seniors (20-45y)', 'Veterans (45y+)']
        exp_groups = pd.cut(unique_active_docs['spec_experience'], bins=bins, labels=labels, right=False)
        exp_counts = exp_groups.value_counts().sort_index()
        pie_labels = exp_counts.index.tolist()
        pie_values = exp_counts.values.tolist()
        
        # CHART 4: Density
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
                "pie_labels": pie_labels, 
                "pie_values": pie_values,
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
        .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; }}
        .chart-box {{ background: white; padding: 10px; border-radius: 8px; border: 1px solid #eee; min-height: 350px; }}
        .chart-full {{ grid-column: 1 / -1; }}
        .footer {{ text-align: center; margin-top: 50px; color: #bdc3c7; font-size: 0.8em; }}
    </style>
</head>
<body>

<div class="container">
    <h1>🇮🇱 Israel Medical Workforce Analysis</h1>
    <div class="subtitle">Active Doctors (Under 43 years experience) | Data Source: MoH Live API</div>

    <div class="section-title">🗺️ Global Market Velocity Map</div>
    <div id="chart-velocity" class="chart-box" style="height: 500px;"></div>

    <div class="section-title">🔬 Specialty Deep Dive</div>
    <div class="controls">
        <label for="specSelect"><strong>Select Specialty: </strong></label>
        <select id="specSelect" onchange="updateDashboard()"></select>
    </div>

    <div class="kpi-row">
        <div class="kpi-card">
            <span class="kpi-val" id="kpi-total">-</span>
            <span class="kpi-label">Active Doctors</span>
        </div>
        <div class="kpi-card">
            <span class="kpi-val" id="kpi-ratio">-</span>
            <span class="kpi-label">Junior/Vet Ratio</span>
        </div>
        <div class="kpi-card">
            <span class="kpi-val" id="kpi-usa">-</span>
            <span class="kpi-label">USA Benchmark Gap</span>
        </div>
    </div>

    <div class="charts-grid">
        <div id="chart-joins" class="chart-box chart-full"></div>
        <div id="chart-trend" class="chart-box chart-full"></div>
        <div id="chart-exp" class="chart-box"></div>
        <div id="chart-dens" class="chart-box"></div>
    </div>
    
    <div class="footer">Last Updated: {TIMESTAMP}</div>
</div>

<script>
    const data = {json_dashboard};
    const globalData = {json_global};
    const specialties = Object.keys(data).sort();

    const mapTrace = {{
        x: globalData.map(d => d.x),
        y: globalData.map(d => d.y),
        text: globalData.map(d => d.name),
        mode: 'markers',
        marker: {{
            size: globalData.map(d => Math.sqrt(d.x) * 1.5),
            color: globalData.map(d => d.y),
            colorscale: 'Viridis',
            showscale: true,
            opacity: 0.8
        }}
    }};
    
    Plotly.newPlot('chart-velocity', [mapTrace], {{
        title: 'Size (Total Doctors) vs Growth Velocity (% Juniors)',
        xaxis: {{ title: 'Total Doctors (Size)' }},
        yaxis: {{ title: 'Velocity (Junior %) - Higher is Faster Growth' }},
        hovermode: 'closest'
    }}, {{responsive: true}});

    const select = document.getElementById('specSelect');
    specialties.forEach(spec => {{
        const opt = document.createElement('option');
        opt.value = spec;
        opt.innerHTML = spec;
        select.appendChild(opt);
    }});

    function updateDashboard() {{
        const spec = select.value;
        const d = data[spec];
        
        document.getElementById('kpi-total').innerText = d.total;
        
        const ratioElem = document.getElementById('kpi-ratio');
        ratioElem.innerText = d.ratio_val;
        ratioElem.style.color = d.ratio_color;
        
        const usaElem = document.getElementById('kpi-usa');
        usaElem.innerText = d.usa_text;
        usaElem.style.color = d.usa_color;

        Plotly.newPlot('chart-joins', [{{
            x: d.charts.years_x,
            y: d.charts.years_y,
            type: 'bar',
            marker: {{ color: '#3498db' }}
        }}], {{
            title: 'New Specialty Licenses Per Year',
            margin: {{ t: 40, b: 40, l: 40, r: 20 }},
            xaxis: {{ title: 'Year' }}
        }}, {{responsive: true}});

        const traceHist = {{
            x: d.charts.hist_x,
            y: d.charts.hist_y,
            name: 'Historical Data',
            type: 'scatter',
            mode: 'lines',
            line: {{ width: 3, color: '#2c3e50' }}
        }};
        
        const traceFut = {{
            x: [d.charts.hist_x[d.charts.hist_x.length-1], ...d.charts.fut_x],
            y: [d.charts.hist_y[d.charts.hist_y.length-1], ...d.charts.fut_y],
            name: 'Projected Forecast',
            type: 'scatter',
            mode: 'lines',
            line: {{ width: 3, color: '#e74c3c', dash: 'dot' }}
        }};

        Plotly.newPlot('chart-trend', [traceHist, traceFut], {{
            title: 'Net Pipeline (Inflow vs 45y Retirement)',
            margin: {{ t: 40, b: 40, l: 40, r: 20 }},
            shapes: [{{ type: 'line', x0: 1980, x1: 2035, y0: 0, y1: 0, line: {{ color: 'gray', width: 1, dash: 'dot' }} }}],
            xaxis: {{ title: 'Year', range: [1980, 2035] }},
            yaxis: {{ title: 'Net Balance (In - Out)' }}
        }}, {{responsive: true}});

        // PIE CHART
        var pieData = [{{
            values: d.charts.pie_values,
            labels: d.charts.pie_labels,
            type: 'pie',
            marker: {{ colors: ['#2ecc71', '#3498db', '#f1c40f', '#e74c3c'] }},
            textinfo: 'label+percent',
            hoverinfo: 'label+value'
        }}];
        
        Plotly.newPlot('chart-exp', pieData, {{
            title: 'Experience Structure (Specialty Date based)',
            margin: {{ t: 40, b: 40, l: 40, r: 40 }}
        }}, {{responsive: true}});

        const densityTrace = {{
            x: d.charts.dens_x,
            y: d.charts.dens_y,
            type: 'bar',
            orientation: 'h',
            marker: {{ color: d.charts.dens_c }},
            text: d.charts.dens_x.map(v => v.toFixed(3)),
            textposition: 'auto'
        }};
        const densLayout = {{
            title: 'Density (per 1,000)',
            margin: {{ t: 40, b: 40, l: 80, r: 20 }},
            xaxis: {{ zeroline: false }}
        }};
        if(d.charts.usa_bench) {{
            densLayout.shapes = [{{ type: 'line', x0: d.charts.usa_bench, x1: d.charts.usa_bench, y0: -0.5, y1: 1.5, line: {{ color: 'red', width: 2, dash: 'dot' }} }}];
        }}
        Plotly.newPlot('chart-dens', [densityTrace], densLayout, {{responsive: true}});
    }}
    
    if (specialties.length > 0) updateDashboard();
</script>

</body>
</html>
    """

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print("✅ Success! Dashboard generated.")

if __name__ == "__main__":
    generate_static_site()
