import pandas as pd
import plotly.express as px
import plotly.io as pio
import datetime

# --- CONFIG ---
CSV_FILE = "israel_doctors_safe.csv"
ISRAEL_POPULATION = 10_170_000
RETIREMENT_AGE_EXPERIENCE = 43

# --- USA BENCHMARKS ---
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

def load_data():
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print("Error: CSV not found.")
        return None

    # Normalization
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
    df['year'] = df['clean_date'].dt.year
    df['experience'] = datetime.datetime.now().year - df['year']
    return df

def generate_html_report():
    df = load_data()
    if df is None: return

    active_df = df[df['experience'] <= RETIREMENT_AGE_EXPERIENCE].copy()
    
    s1 = active_df['specialty_1'].dropna().astype(str).unique().tolist()
    s2 = active_df['specialty_2'].dropna().astype(str).unique().tolist()
    unique_specialties = sorted(list(set(s1 + s2)))
    unique_specialties = [s for s in unique_specialties if s.lower() not in ['nan', 'none', '']]

    # --- CALCULATE STATS ---
    all_stats = []
    for spec in unique_specialties:
        mask = (active_df['specialty_1'] == spec) | (active_df['specialty_2'] == spec)
        spec_df = active_df[mask]
        total = len(spec_df)
        if total < 30: continue

        inflow = len(spec_df[spec_df['experience'] <= 10])
        outflow = len(spec_df[spec_df['experience'] >= (RETIREMENT_AGE_EXPERIENCE - 10)])
        net_change = inflow - outflow
        velocity = (inflow / total) * 100
        israel_density = (total / ISRAEL_POPULATION) * 1000
        
        bench = AAMC_USA_BENCHMARKS.get(spec)
        gap_str = "N/A"
        gap_color = "gray"
        
        if bench:
            gap = israel_density - bench
            gap_docs = int(gap * (ISRAEL_POPULATION / 1000))
            if gap < 0:
                gap_str = f"🔴 Shortage ({gap_docs})"
                gap_color = "#ffebee" # Light Red
            else:
                gap_str = f"🟢 Surplus (+{gap_docs})"
                gap_color = "#e8f5e9" # Light Green
        
        trend_icon = "🟢" if net_change > 0 else "🔴"
        if abs(net_change) < 5: trend_icon = "🟡"

        all_stats.append({
            'Specialty': spec,
            'Total Doctors': total,
            'Net Change (10y)': f"{trend_icon} {net_change:+d}",
            'Velocity (% Juniors)': f"{velocity:.1f}%",
            'USA Status': gap_str,
            '_raw_velocity': velocity
        })

    stats_df = pd.DataFrame(all_stats)
    stats_df = stats_df.sort_values('_raw_velocity', ascending=False).drop(columns=['_raw_velocity'])

    # --- 1. CREATE PLOTLY CHART ---
    fig = px.scatter(
        stats_df, x='Total Doctors', y=stats_df['Velocity (% Juniors)'].str.replace('%','').astype(float),
        hover_name='Specialty', size='Total Doctors', color='USA Status',
        title="<b>Market Map:</b> Size vs Growth Velocity",
        height=500
    )
    fig.update_layout(template="plotly_white")
    chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

    # --- 2. CREATE HTML TABLE ---
    # We use simple CSS to make the table look professional
    table_html = stats_df.to_html(index=False, classes='styled-table', escape=False)

    # --- 3. ASSEMBLE FULL HTML PAGE ---
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Israel Medical Workforce Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f9; }}
            .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
            h1 {{ text-align: center; color: #333; }}
            .summary-box {{ background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin-bottom: 20px; text-align: center; }}
            
            /* Table Styling */
            .styled-table {{ width: 100%; border-collapse: collapse; margin: 25px 0; font-size: 0.9em; box-shadow: 0 0 20px rgba(0, 0, 0, 0.15); }}
            .styled-table thead tr {{ background-color: #009879; color: #ffffff; text-align: left; }}
            .styled-table th, .styled-table td {{ padding: 12px 15px; border-bottom: 1px solid #dddddd; }}
            .styled-table tbody tr:nth-of-type(even) {{ background-color: #f3f3f3; }}
            .styled-table tbody tr:last-of-type {{ border-bottom: 2px solid #009879; }}
            
            /* Responsive */
            @media screen and (max-width: 600px) {{
                .styled-table {{ display: block; overflow-x: auto; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🇮🇱 Israel Medical Workforce Report</h1>
            <div class="summary-box">
                <p><strong>Generated on:</strong> {datetime.datetime.now().strftime('%Y-%m-%d')}</p>
                <p>Data based on Ministry of Health Registry (Active Doctors &lt; {RETIREMENT_AGE_EXPERIENCE} years exp)</p>
            </div>

            {chart_html}

            <hr style="margin: 40px 0;">

            <h2>Detailed Data Table</h2>
            {table_html}
            
            <p style="text-align:center; color:gray; font-size:12px; margin-top:50px;">
                Generated by Python Analysis
            </p>
        </div>
    </body>
    </html>
    """

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(full_html)
    
    print("✅ Success! 'index.html' has been created.")

if __name__ == "__main__":
    generate_html_report()