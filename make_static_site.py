import pandas as pd
import json
import datetime
import plotly.utils

# --- CONFIGURATION ---
CSV_FILE = "israel_doctors_safe.csv"
ISRAEL_POPULATION = 10_170_000
RETIREMENT_AGE_EXPERIENCE = 45

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
        print("❌ Error: CSV file not found.")
        return None

    # --- NORMALIZATION ---
    ent_target = 'מחלות אף אוזן וגרון'
    ent_source = 'מחלות א.א.ג. וכירורגיית ראש-צוואר'
    df.loc[df['specialty_1'] == ent_source, 'specialty_1'] = ent_target
    df.loc[df['specialty_2'] == ent_source, 'specialty_2'] = ent_target

    thoracic_target = 'כירורגיה חזה ולב'
    pattern = 'חזה|לב'
    mask1 = df['specialty_1'].astype(str).str.contains(pattern, regex=True, na=False)
    df.loc[mask1, 'specialty_1'] = thoracic_target
    mask2 = df['specialty_2'].astype(str).str.contains(pattern, regex=True, na=False)
    df.loc[mask2, 'specialty_2'] = thoracic_target

    normalization_map = {
        'רפואת משפחה': 'רפואת המשפחה', 'אורתופדיה': 'כירורגיה אורתופדית',
        'עיניים': 'מחלות עיניים', 'רפואת עיניים': 'מחלות עיניים',
        'אורולוגיה': 'כירורגיה אורולוגית', 'עור ומין': 'דרמטולוגיה-מחלות עור ומין',
        'כירורגיה פלסטית': 'כירורגיה פלסטית ואסתטית', 'טיפול נמרץ': 'טיפול נמרץ כללי'
    }
    df['specialty_1'] = df['specialty_1'].replace(normalization_map)
    df['specialty_2'] = df['specialty_2'].replace(normalization_map)

    df['clean_date'] = pd.to_datetime(df['registration_date'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['clean_date'])
    df['reg_year'] = df['clean_date'].dt.year
    df['retirement_year'] = df['reg_year'] + RETIREMENT_AGE_EXPERIENCE
    
    current_year = datetime.datetime.now().year
    df['experience'] = current_year - df['reg_year']
    
    return df

def generate_static_site():
    df = load_and_clean_data()
    if df is None: return

    # Filter Active Doctors
    active_df = df[df['experience'] <= RETIREMENT_AGE_EXPERIENCE].copy()
    
    # Get Unique Specialties
    s1 = active_df['specialty_1'].dropna().astype(str).unique().tolist()
    s2 = active_df['specialty_2'].dropna().astype(str).unique().tolist()
    unique_specialties = sorted(list(set(s1 + s2)))
    unique_specialties = [s for s in unique_specialties if s.lower() not in ['nan', 'none', '']]

    # --- PRE-CALCULATE ALL DATA ---
    dashboard_data = {}
    
    print("⏳ Processing specialties...")
    
    for spec in unique_specialties:
        # LOGIC: Spec 1 OR Spec 2
        mask = (active_df['specialty_1'] == spec) | (active_df['specialty_2'] == spec)
        spec_df = active_df[mask]
        total = len(spec_df)
        
        if total < 30: continue # Skip small specialties

        # 1. HEADLINE STATS (Current Snapshot)
        inflow_now = len(spec_df[spec_df['experience'] <= 10])
        outflow_now = len(spec_df[spec_df['experience'] >= (RETIREMENT_AGE_EXPERIENCE - 10)])
        net_now = inflow_now - outflow_now
        
        density = (total / ISRAEL_POPULATION) * 1000
        usa_bench = AAMC_USA_BENCHMARKS.get(spec, None)
        
        usa_text = "No Benchmark"
        usa_color = "gray"
        
        if usa_bench:
            gap = density - usa_bench
            gap_docs = int(gap * (ISRAEL_POPULATION / 1000))
            if gap < 0:
                usa_text = f"Deficit: {gap_docs} docs"
                usa_color = "#e74c3c" # Red
            else:
                usa_text = f"Surplus: +{gap_docs} docs"
                usa_color = "#2ecc71" # Green

        # 2. EXPERIENCE STRUCTURE (Histogram)
        bins = [0, 10, 20, 30, 40]
        labels = ['Juniors (0-10)', 'Mid (10-20)', 'Senior (20-30)', 'Vet (30+)']
        exp_groups = pd.cut(spec_df['experience'], bins=bins, labels=labels, right=False)
        exp_counts = exp_groups.value_counts().sort_index().tolist()
        
        # 3. TRAILING NET CHANGE (Time Series Analysis)
        # We calculate the window for years 2015 to 2030
        years_range = list(range(2015, 2031))
        net_trend_y = []
        
        for y in years_range:
            # Inflow: Registered between (Y-10) and Y
            inflow_y = len(spec_df[(spec_df['reg_year'] > (y - 10)) & (spec_df['reg_year'] <= y)])
            
            # Outflow: Retiring between Y and (Y+10)
            # Retiring year is reg_year + 43
            outflow_y = len(spec_df[(spec_df['retirement_year'] >= y) & (spec_df['retirement_year'] < (y + 10))])
            
            net_trend_y.append(inflow_y - outflow_y)

        # 4. DENSITY COMPARISON
        density_x = [density]
        density_y = ['Israel']
        density_colors = ['#3498db']
        
        if usa_bench:
            density_x.append(usa_bench)
            density_y.append('USA')
            density_colors.append('#2c3e50')

        # STORE DATA
        dashboard_data[spec] = {
            "total": int(total),
            "net_now": int(net_now),
            "usa_text": usa_text,
            "usa_color": usa_color,
            "charts": {
                "exp_x": labels,
                "exp_y": exp_counts,
                "trend_x": years_range,
                "trend_y": net_trend_y,
                "dens_x": density_x,
                "dens_y": density_y,
                "dens_c": density_colors,
                "usa_bench": usa_bench  # For drawing the line
            }
        }

    # SERIALIZE TO JSON
    json_data = json.dumps(dashboard_data, default=lambda x: int(x) if isinstance(x, (np.int64, np.int32)) else x)

    # --- HTML TEMPLATE ---
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
        
        .controls {{ text-align: center; margin-bottom: 30px; padding: 20px; background: #ecf0f1; border-radius: 8px; }}
        select {{ padding: 10px 20px; font-size: 16px; border-radius: 5px; border: 1px solid #bdc3c7; min-width: 300px; }}
        
        .kpi-row {{ display: flex; justify-content: space-around; margin-bottom: 30px; flex-wrap: wrap; gap: 10px; }}
        .kpi-card {{ background: #fff; padding: 15px 25px; border-radius: 8px; border: 1px solid #eee; text-align: center; min-width: 200px; box-shadow: 0 2px 5px rgba(0,0,0,0.03); }}
        .kpi-val {{ font-size: 24px; font-weight: bold; color: #2c3e50; display: block; }}
        .kpi-label {{ font-size: 14px; color: #95a5a6; text-transform: uppercase; letter-spacing: 1px; }}
        
        .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; }}
        .chart-box {{ background: white; padding: 10px; border-radius: 8px; border: 1px solid #eee; min-height: 350px; }}
    </style>
</head>
<body>

<div class="container">
    <h1>🇮🇱 Israel Medical Workforce Analysis</h1>
    <div class="subtitle">Data source: Ministry of Health Registry (Active Doctors &lt; 43y experience)</div>

    <div class="controls">
        <label for="specSelect"><strong>Select Specialty: </strong></label>
        <select id="specSelect" onchange="updateDashboard()"></select>
    </div>

    <div class="kpi-row">
        <div class="kpi-card">
            <span class="kpi-val" id="kpi-total">-</span>
            <span class="kpi-label">Total Doctors</span>
        </div>
        <div class="kpi-card">
            <span class="kpi-val" id="kpi-net">-</span>
            <span class="kpi-label">Current Net Trend</span>
        </div>
        <div class="kpi-card">
            <span class="kpi-val" id="kpi-usa">-</span>
            <span class="kpi-label">USA Benchmark Gap</span>
        </div>
    </div>

    <div class="charts-grid">
        <div id="chart-exp" class="chart-box"></div>
        <div id="chart-trend" class="chart-box"></div>
        <div id="chart-dens" class="chart-box"></div>
    </div>
</div>

<script>
    // --- EMBEDDED DATA ---
    const data = {json_data};
    const specialties = Object.keys(data).sort();

    // --- INITIALIZE DROPDOWN ---
    const select = document.getElementById('specSelect');
    specialties.forEach(spec => {{
        const opt = document.createElement('option');
        opt.value = spec;
        opt.innerHTML = spec;
        select.appendChild(opt);
    }});

    // --- MAIN UPDATE FUNCTION ---
    function updateDashboard() {{
        const spec = select.value;
        const d = data[spec];
        
        // 1. Update KPIs
        document.getElementById('kpi-total').innerText = d.total;
        
        const netElem = document.getElementById('kpi-net');
        netElem.innerText = (d.net_now > 0 ? "+" : "") + d.net_now;
        netElem.style.color = d.net_now >= 0 ? "#2ecc71" : "#e74c3c";

        const usaElem = document.getElementById('kpi-usa');
        usaElem.innerText = d.usa_text;
        usaElem.style.color = d.usa_color;

        // 2. PLOT: Experience Structure (Bar)
        Plotly.newPlot('chart-exp', [{{
            x: d.charts.exp_x,
            y: d.charts.exp_y,
            type: 'bar',
            marker: {{ color: ['#2ecc71', '#3498db', '#3498db', '#e74c3c'] }}
        }}], {{
            title: '1. Experience Structure',
            margin: {{ t: 40, b: 40, l: 40, r: 20 }},
            yaxis: {{ title: 'Number of Doctors' }}
        }}, {{responsive: true}});

        // 3. PLOT: Trailing Net Change (Line)
        // Color line green if above 0, red if below
        const trendTrace = {{
            x: d.charts.trend_x,
            y: d.charts.trend_y,
            type: 'scatter',
            mode: 'lines+markers',
            line: {{ width: 3, color: '#34495e' }},
            marker: {{ 
                color: d.charts.trend_y.map(v => v >= 0 ? '#2ecc71' : '#e74c3c'),
                size: 8
            }}
        }};
        
        Plotly.newPlot('chart-trend', [trendTrace], {{
            title: '2. Net Change Trend (10y Window)',
            margin: {{ t: 40, b: 40, l: 40, r: 20 }},
            shapes: [{{ type: 'line', x0: d.charts.trend_x[0], x1: d.charts.trend_x[d.charts.trend_x.length-1], y0: 0, y1: 0, line: {{ color: 'gray', width: 1, dash: 'dot' }} }}],
            xaxis: {{ title: 'Year' }},
            yaxis: {{ title: 'Net Change (In - Out)' }}
        }}, {{responsive: true}});

        // 4. PLOT: Density vs USA (Horizontal Bar)
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
            title: '3. Density (per 1,000)',
            margin: {{ t: 40, b: 40, l: 80, r: 20 }},
            xaxis: {{ zeroline: false }}
        }};

        // Add benchmark line if USA exists
        if(d.charts.usa_bench) {{
            densLayout.shapes = [{{
                type: 'line',
                x0: d.charts.usa_bench, x1: d.charts.usa_bench,
                y0: -0.5, y1: 1.5,
                line: {{ color: 'red', width: 2, dash: 'dot' }}
            }}];
        }}

        Plotly.newPlot('chart-dens', [densityTrace], densLayout, {{responsive: true}});
    }}

    // Initialize first view
    updateDashboard();
</script>

</body>
</html>
    """

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print("✅ Success! 'index.html' updated with new interactive logic.")

if __name__ == "__main__":
    generate_static_site()
