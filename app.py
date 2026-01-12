# app.py - Deploy this to Render, Railway, or PythonAnywhere

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Generate sample data (in production, load from actual source)
np.random.seed(42)
n_customers = 7043

data = {
    'customerID': [f'CUST{i:04d}' for i in range(n_customers)],
    'tenure': np.random.randint(0, 73, n_customers),
    'MonthlyCharges': np.random.uniform(20, 120, n_customers),
    'TotalCharges': np.random.uniform(100, 8000, n_customers),
    'Contract': np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], n_customers, p=[0.55, 0.21, 0.24]),
    'PaymentMethod': np.random.choice(['Electronic Check', 'Mailed Check', 'Bank Transfer', 'Credit Card'], n_customers),
    'InternetService': np.random.choice(['DSL', 'Fiber Optic', 'No'], n_customers, p=[0.35, 0.45, 0.20]),
    'Churn': np.random.choice([0, 1], n_customers, p=[0.735, 0.265])
}

df = pd.DataFrame(data)

# Create tenure groups
df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 6, 12, 24, 48, 73], 
                            labels=['0-6m', '6-12m', '12-24m', '24-48m', '48+m'])

# Create clusters (simplified for demo)
conditions = [
    (df['tenure'] < 12) & (df['Contract'] == 'Month-to-Month'),
    (df['tenure'].between(12, 36)) | (df['Contract'] == 'One Year'),
    (df['tenure'] > 36) | (df['Contract'] == 'Two Year')
]
df['cluster'] = np.select(conditions, [2, 1, 0], default=1)  # 2=High Risk, 1=Medium, 0=Low

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # For deployment

# Layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🎯 Customer Churn Analysis Dashboard", 
                   className="text-center text-primary mb-3 mt-4"),
            html.P("Real-time insights and predictive analytics for customer retention", 
                  className="text-center text-muted mb-4")
        ])
    ]),
    
    html.Hr(),
    
    # Filters Section
    dbc.Card([
        dbc.CardBody([
            html.H5("📋 Filters", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Contract Type:", className="font-weight-bold"),
                    dcc.Dropdown(
                        id='contract-filter',
                        options=[{'label': 'All Contracts', 'value': 'all'}] + 
                                [{'label': c, 'value': c} for c in df['Contract'].unique()],
                        value='all',
                        clearable=False,
                        style={'width': '100%'}
                    )
                ], md=4),
                
                dbc.Col([
                    html.Label("Risk Segment:", className="font-weight-bold"),
                    dcc.Dropdown(
                        id='segment-filter',
                        options=[
                            {'label': 'All Segments', 'value': 'all'},
                            {'label': '🔴 High Risk', 'value': 2},
                            {'label': '🟡 Medium Risk', 'value': 1},
                            {'label': '🟢 Low Risk', 'value': 0}
                        ],
                        value='all',
                        clearable=False
                    )
                ], md=4),
                
                dbc.Col([
                    html.Label("Internet Service:", className="font-weight-bold"),
                    dcc.Dropdown(
                        id='internet-filter',
                        options=[{'label': 'All Services', 'value': 'all'}] + 
                                [{'label': s, 'value': s} for s in df['InternetService'].unique()],
                        value='all',
                        clearable=False
                    )
                ], md=4)
            ])
        ])
    ], className="mb-4 shadow-sm"),
    
    # KPI Cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H6("📉 Churn Rate", className="text-muted mb-2"),
                        html.H2(id="kpi-churn-rate", className="text-danger mb-1"),
                        html.Small(id="kpi-churn-count", className="text-muted")
                    ])
                ])
            ], className="shadow-sm h-100")
        ], md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H6("👥 Total Customers", className="text-muted mb-2"),
                        html.H2(id="kpi-total-customers", className="text-primary mb-1"),
                        html.Small("In filtered segment", className="text-muted")
                    ])
                ])
            ], className="shadow-sm h-100")
        ], md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H6("💰 Avg Monthly Charges", className="text-muted mb-2"),
                        html.H2(id="kpi-avg-charges", className="text-success mb-1"),
                        html.Small(id="kpi-total-revenue", className="text-muted")
                    ])
                ])
            ], className="shadow-sm h-100")
        ], md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H6("⚠️ High Risk Customers", className="text-muted mb-2"),
                        html.H2(id="kpi-at-risk", className="text-warning mb-1"),
                        html.Small("Immediate attention needed", className="text-muted")
                    ])
                ])
            ], className="shadow-sm h-100")
        ], md=3)
    ], className="mb-4"),
    
    # Main Charts
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("📊 Churn Rate by Contract Type", className="mb-0")),
                dbc.CardBody([
                    dcc.Graph(id='churn-by-contract', config={'displayModeBar': False})
                ])
            ], className="shadow-sm h-100")
        ], md=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("📈 Monthly Charges Distribution", className="mb-0")),
                dbc.CardBody([
                    dcc.Graph(id='charges-distribution', config={'displayModeBar': False})
                ])
            ], className="shadow-sm h-100")
        ], md=6)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("🎯 Customer Segments Analysis", className="mb-0")),
                dbc.CardBody([
                    dcc.Graph(id='segment-scatter', config={'displayModeBar': False})
                ])
            ], className="shadow-sm h-100")
        ], md=8),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("🔄 Payment Method Impact", className="mb-0")),
                dbc.CardBody([
                    dcc.Graph(id='payment-pie', config={'displayModeBar': False})
                ])
            ], className="shadow-sm h-100")
        ], md=4)
    ], className="mb-4"),
    
    # Recommendations
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("💡 Key Recommendations", className="mb-0")),
                dbc.CardBody([
                    dbc.Alert([
                        html.H6("🎯 Priority Actions:", className="alert-heading mb-3"),
                        html.Ul([
                            html.Li("Focus retention on month-to-month contracts (highest churn risk)"),
                            html.Li("Implement enhanced onboarding for customers in first 6 months"),
                            html.Li("Incentivize annual/bi-annual contract upgrades with discounts"),
                            html.Li("Bundle essential services (tech support, online security) for high-risk segments"),
                            html.Li("Promote automatic payment methods to reduce churn by 30%")
                        ], className="mb-0")
                    ], color="info", className="mb-0")
                ])
            ], className="shadow-sm")
        ])
    ], className="mb-4"),
    
    # Footer
    html.Hr(),
    html.P("© 2026 Customer Churn Analysis | Powered by Python & Dash", 
           className="text-center text-muted mb-4")
    
], fluid=True, style={'backgroundColor': '#f8f9fa'})


# Callbacks
@app.callback(
    [Output('kpi-churn-rate', 'children'),
     Output('kpi-churn-count', 'children'),
     Output('kpi-total-customers', 'children'),
     Output('kpi-avg-charges', 'children'),
     Output('kpi-total-revenue', 'children'),
     Output('kpi-at-risk', 'children')],
    [Input('contract-filter', 'value'),
     Input('segment-filter', 'value'),
     Input('internet-filter', 'value')]
)
def update_kpis(contract, segment, internet):
    filtered_df = df.copy()
    
    if contract != 'all':
        filtered_df = filtered_df[filtered_df['Contract'] == contract]
    if segment != 'all':
        filtered_df = filtered_df[filtered_df['cluster'] == segment]
    if internet != 'all':
        filtered_df = filtered_df[filtered_df['InternetService'] == internet]
    
    churn_rate = f"{filtered_df['Churn'].mean()*100:.1f}%"
    churn_count = f"{filtered_df['Churn'].sum():,} of {len(filtered_df):,} customers"
    total_customers = f"{len(filtered_df):,}"
    avg_charges = f"${filtered_df['MonthlyCharges'].mean():.2f}"
    total_revenue = f"Total: ${filtered_df['MonthlyCharges'].sum():,.0f}/month"
    at_risk = f"{(filtered_df['cluster']==2).sum():,}"
    
    return churn_rate, churn_count, total_customers, avg_charges, total_revenue, at_risk


@app.callback(
    Output('churn-by-contract', 'figure'),
    [Input('contract-filter', 'value'),
     Input('segment-filter', 'value'),
     Input('internet-filter', 'value')]
)
def update_contract_chart(contract, segment, internet):
    filtered_df = df.copy()
    
    if contract != 'all':
        filtered_df = filtered_df[filtered_df['Contract'] == contract]
    if segment != 'all':
        filtered_df = filtered_df[filtered_df['cluster'] == segment]
    if internet != 'all':
        filtered_df = filtered_df[filtered_df['InternetService'] == internet]
    
    contract_churn = filtered_df.groupby(['Contract', 'Churn']).size().reset_index(name='count')
    
    fig = px.bar(
        contract_churn,
        x='Contract',
        y='count',
        color='Churn',
        barmode='group',
        labels={'count': 'Number of Customers', 'Churn': 'Status'},
        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
        height=300
    )
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(title='', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig


@app.callback(
    Output('charges-distribution', 'figure'),
    [Input('contract-filter', 'value'),
     Input('segment-filter', 'value'),
     Input('internet-filter', 'value')]
)
def update_charges_chart(contract, segment, internet):
    filtered_df = df.copy()
    
    if contract != 'all':
        filtered_df = filtered_df[filtered_df['Contract'] == contract]
    if segment != 'all':
        filtered_df = filtered_df[filtered_df['cluster'] == segment]
    if internet != 'all':
        filtered_df = filtered_df[filtered_df['InternetService'] == internet]
    
    fig = px.box(
        filtered_df,
        x='Churn',
        y='MonthlyCharges',
        color='Churn',
        labels={'Churn': 'Customer Status', 'MonthlyCharges': 'Monthly Charges ($)'},
        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
        height=300
    )
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False
    )
    
    return fig


@app.callback(
    Output('segment-scatter', 'figure'),
    [Input('contract-filter', 'value'),
     Input('segment-filter', 'value'),
     Input('internet-filter', 'value')]
)
def update_segment_scatter(contract, segment, internet):
    filtered_df = df.copy()
    
    if contract != 'all':
        filtered_df = filtered_df[filtered_df['Contract'] == contract]
    if segment != 'all':
        filtered_df = filtered_df[filtered_df['cluster'] == segment]
    if internet != 'all':
        filtered_df = filtered_df[filtered_df['InternetService'] == internet]
    
    cluster_labels = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
    filtered_df['Risk_Level'] = filtered_df['cluster'].map(cluster_labels)
    
    fig = px.scatter(
        filtered_df,
        x='tenure',
        y='MonthlyCharges',
        color='Risk_Level',
        size='TotalCharges',
        labels={'tenure': 'Tenure (months)', 'MonthlyCharges': 'Monthly Charges ($)', 'Risk_Level': 'Risk Level'},
        color_discrete_map={'Low Risk': '#2ecc71', 'Medium Risk': '#f39c12', 'High Risk': '#e74c3c'},
        height=400,
        hover_data=['TotalCharges']
    )
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(title='Risk Level', orientation='v', yanchor='top', y=1, xanchor='right', x=1)
    )
    
    return fig


@app.callback(
    Output('payment-pie', 'figure'),
    [Input('contract-filter', 'value'),
     Input('segment-filter', 'value'),
     Input('internet-filter', 'value')]
)
def update_payment_pie(contract, segment, internet):
    filtered_df = df.copy()
    
    if contract != 'all':
        filtered_df = filtered_df[filtered_df['Contract'] == contract]
    if segment != 'all':
        filtered_df = filtered_df[filtered_df['cluster'] == segment]
    if internet != 'all':
        filtered_df = filtered_df[filtered_df['InternetService'] == internet]
    
    payment_churn = filtered_df.groupby('PaymentMethod')['Churn'].mean().reset_index()
    payment_churn['Churn_Rate'] = payment_churn['Churn'] * 100
    
    fig = px.pie(
        payment_churn,
        values='Churn_Rate',
        names='PaymentMethod',
        title='',
        height=400,
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    
    return fig


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
