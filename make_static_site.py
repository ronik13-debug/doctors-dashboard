import pandas as pd
import json
import datetime
import plotly.utils
import numpy as np
import os

# --- CONFIGURATION ---
CSV_FILE = "israel_doctors.csv" # Ensure this matches your filename
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
    if not os.path.exists(CSV_FILE):
        print(f"❌ Error: {CSV_FILE} not found. Please run the scraper first.")
        return None

    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return None

    # --- 1. NORMALIZE SPECIALTY NAMES ---
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

    # --- 2. PARSE DATES ---
    
    # A. General Registration Date (Column B) -> Used for "Active Doctor" filtering & Experience Grouping
    df['gen_date'] = pd.to_datetime(df['Registration Date'], format='%d/%m/%Y', errors='coerce')
    
    # B. Specialty 1 Date (Column D)
    df['s1_date'] = pd.to_datetime(df['Specialty 1 Date'], format='%d/%m/%Y', errors='coerce')
    
    # C. Specialty 2 Date (Column F)
    df['s2_date'] = pd.to_datetime(df['Specialty 2 Date'], format='%d/%m/%Y', errors='coerce')

    # Drop rows without a valid general registration date
    df = df.dropna(subset=['gen_date'])

    # --- 3. CALCULATE GENERAL EXPERIENCE ---
    df['gen_year'] = df['gen_date'].dt.year
    df['gen_experience'] = CURRENT_YEAR - df['gen_year']
    
    return df

def generate_static_site():
    df = load_and_clean_data()
    if df is None: return

    col_s1_name = 'Specialty 1 Name'
    col_s2_name = 'Specialty 2 Name'

    # Filter Active Doctors based on GENERAL EXPERIENCE (Column B)
    # This keeps the dashboard focused on doctors who are actually working age
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
        mask_s1_all = df[col_s1_name] == spec
        mask_s2_all = df[col_s2_name] == spec
        
        mask_s1_active = active_df[col_s1_name] == spec
        mask_s2_active = active_df[col_s2_name] == spec
        spec_df_active = active_df[mask_s1_active | mask_s2_active].copy()
        
        total_active = len(spec_df_active)
        if total_active < 30: continue 

        # --- 2. CONSTRUCT SPECIALTY-SPECIFIC TIMELINES ---
        # Get the specific date when each doctor acquired THIS specialty
        dates_from_s1 = df.loc[mask_s1_all, 's1_date']
        dates_from_s2 = df.loc[mask_s2_all, 's2_date']
        all_spec_dates = pd.concat([dates_from_s1, dates_from_s2])
        
        # Determine the "Specialty Start Year" for everyone
        spec_start_years = all_spec_dates.dt.year
        
        # Determine "Specialty Retirement Year" (Specialty Start + 45)
        # We use this for the Net Pipeline Outflow calculation
        spec_retirement_years = spec_start_years + RETIREMENT_AGE_EXPERIENCE

        # --- 3. REPLACEMENT RATIO (KPI) ---
        # Based on General Experience (Col B) to reflect biological age/seniority
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
        
        # Net Now (KPI)
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

        # --- 5. CHARTS LOGIC ---
        
        # CHART 1: New Licenses (Inflow)
        joins_per_year = spec_start_years.value_counts().sort_index()
        years_idx = list(range(1980, CURRENT_YEAR + 1))
        joins_counts = [int(joins_per_year.get(y, 0)) for y in years_idx]

        # Calculate Projected Inflow for Forecast (Avg of last 5 years)
        recent_years = range(CURRENT_YEAR - 5, CURRENT_YEAR)
        recent_inflow_sum = sum([joins_per_year.get(y, 0) for y in recent_years])
        avg_inflow = max(1, int(recent_inflow_sum / 5))

        # CHART 2: Net Pipeline (Inflow - Outflow)
        history_years = list(range(1980, CURRENT_YEAR + 1))
        future_years = list(range(CURRENT_YEAR + 1, 2036))
        
        net_trend_history = []
        net_trend_forecast = []

        # HISTORY (1980-Current)
        for y in history_years:
            # Inflow: Count doctors who got THIS specialty in the last 10 years
            inflow_y = len(spec_start_years[(spec_start_years > (y - 10)) & (spec_start_years <= y)])
            
            # Outflow: Count doctors who reach "Specialty Retirement Age" in next 10 years
            # (Using specialty_date + 45)
            outflow_y = len(spec_retirement_years[(spec_retirement_years >= y) & (spec_retirement_years < (y + 10))])
            
            net_trend_history.append(inflow_y - outflow_y)

        # FORECAST (Future)
        for y in future_years:
            # Forecast Inflow
            real_part_years = range(y - 10, CURRENT_YEAR + 1)
            projected_part_years = range(max(y - 10, CURRENT_YEAR + 1), y + 1)
            
            count_real = len(spec_start_years[spec_start_years.isin(real_part_years)])
            count_proj = len(projected_part_years) * avg_inflow
            inflow_forecast = count_real + count_proj
            
            # Forecast Outflow (Based on Specialty Date)
            outflow_forecast = len(spec_retirement_years[(spec_retirement_years >= y) & (spec_retirement_years < (y + 10))])
            
            net_trend_forecast.append(inflow_forecast - outflow_forecast)

        # CHART 3: Experience Structure (Keep General Experience for this)
        bins = [0, 10, 20, 30, 40]
        labels = ['Juniors (0-10)', 'Mid (10-20)', 'Senior (20-30)', 'Vet (30+)']
        exp_groups = pd.cut(spec_df_active['gen_experience'], bins=bins, labels=labels, right=False)
        exp_counts = exp_groups.value_counts().sort_index().tolist()
        
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
        .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; }}
        .chart-box {{ background: white; padding: 10px; border-radius: 8px; border: 1px solid #eee; min-height: 350px; }}
        .chart-full {{ grid-column: 1 / -1; }}
        .explanation {{ background: #fef9e7; padding: 15px; margin-top: 20px; border-left: 5px solid #f1c40f; font-size: 0.9em; color: #7f8c8d; }}
    </style>
</head>
<body>

<div class="container">
    <h1>🇮🇱 Israel Medical Workforce Analysis</h1>
    <div class="subtitle">Active Doctors (Under 43 years experience) | Data Source: MoH</div>

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

    <div class="explanation">
        <strong>🔮 How is the "Projected Forecast" (Dotted Line) calculated?</strong><br>
        1. <strong>Future Outflow:</strong> We know exactly when current doctors will retire, so we count them.<br>
        2. <strong>Future Inflow:</strong> We assume the number of new licenses remains similar to the average of the last 5 years.<br>
        This reveals if the current training pipeline is sufficient to handle the upcoming "Retirement Wave".
    </div>
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
            title: 'New Licenses Issued Per Year (1980-{CURRENT_YEAR})',
            margin: {{ t: 40, b: 40, l: 40, r: 20 }},
            xaxis: {{ title: 'Year' }}
        }}, {{responsive: true}});

        // TREND + FORECAST CHART
        const traceHist = {{
            x: d.charts.hist_x,
            y: d.charts.hist_y,
            name: 'Historical Data',
            type: 'scatter',
            mode: 'lines',
            line: {{ width: 3, color: '#2c3e50' }}
        }};
        
        const traceFut = {{
            x: [d.charts.hist_x[d.charts.hist_x.length-1], ...d.charts.fut_x], // Connect lines
            y: [d.charts.hist_y[d.charts.hist_y.length-1], ...d.charts.fut_y],
            name: 'Projected Forecast',
            type: 'scatter',
            mode: 'lines',
            line: {{ width: 3, color: '#e74c3c', dash: 'dot' }}
        }};

        Plotly.newPlot('chart-trend', [traceHist, traceFut], {{
            title: 'Net Pipeline Health & Forecast (1980-2035)',
            margin: {{ t: 40, b: 40, l: 40, r: 20 }},
            shapes: [{{ type: 'line', x0: 1980, x1: 2035, y0: 0, y1: 0, line: {{ color: 'gray', width: 1, dash: 'dot' }} }}],
            xaxis: {{ title: 'Year', range: [1980, 2035] }},
            yaxis: {{ title: 'Net Balance (In - Out)' }}
        }}, {{responsive: true}});

        Plotly.newPlot('chart-exp', [{{
            x: d.charts.exp_x,
            y: d.charts.exp_y,
            type: 'bar',
            marker: {{ color: ['#2ecc71', '#3498db', '#3498db', '#e74c3c'] }}
        }}], {{
            title: 'Experience Structure (Active Doctors)',
            margin: {{ t: 40, b: 40, l: 40, r: 20 }}
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

    // Trigger update for first specialty
    if (specialties.length > 0) {{
       updateDashboard();
    }}
</script>

</body>
</html>
    """

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print("✅ Success! Forecast & Replacement Ratios added.")

if __name__ == "__main__":
    generate_static_site()
