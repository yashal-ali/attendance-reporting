
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time, timedelta
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import io
import re
import os

# Configure the page
st.set_page_config(
    page_title="Advanced Attendance Analytics & ML System",
    page_icon="🤖📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
        font-weight: 500;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2a5298;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #1f77b4;
        background: linear-gradient(90deg, #f8f9fa, #ffffff);
        padding: 1rem;
        border-radius: 8px;
    }
    .analytics-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    .ml-prediction {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .rule-based {
        background: linear-gradient(45deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .tab-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        width:300px;
        margin-right:4px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedAttendanceAnalyticsSystem:
    def __init__(self):
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.processed_data = None
        self.ml_model_loaded = False
        self.feature_columns = []
    
    def load_ml_model(self, model_path):
        """Load the trained ML model"""
        try:
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                self.model = model_data['model']
                self.scaler = model_data.get('scaler', StandardScaler())
                self.label_encoder = model_data.get('label_encoder', LabelEncoder())
                self.feature_columns = model_data.get('feature_columns', [])
                self.ml_model_loaded = True
                return True
            else:
                st.error(f"Model file not found: {model_path}")
                return False
        except Exception as e:
            st.error(f"Error loading ML model: {e}")
            return False
    
    def load_and_process_data(self, file_path):
        """Load and process the main dataset for analytics"""
        try:
            if file_path.endswith('.xlsx'):
                self.df = pd.read_excel(file_path)
            else:
                self.df = pd.read_csv(file_path)
            
            # Process the data
            self.processed_data = self.process_data_for_analytics(self.df)
            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False
    
    def process_data_for_analytics(self, df):
        """Process data for comprehensive analytics"""
        df_processed = df.copy()
        
        # Convert date/time
        df_processed['Attendance Date'] = pd.to_datetime(df_processed['Attendance Date'])
        df_processed['Start Day Time'] = pd.to_datetime(df_processed['Start Day Time']).dt.time
        
        # Create datetime for calculations
        df_processed['checkin_datetime'] = pd.to_datetime(
            df_processed['Attendance Date'].astype(str) + ' ' + 
            df_processed['Start Day Time'].astype(str)
        )
        
        # Calculate minutes late
        reference_time = pd.to_datetime(df_processed['Attendance Date'].astype(str) + ' 09:15:00')
        df_processed['minutes_late'] = (df_processed['checkin_datetime'] - reference_time).dt.total_seconds() / 60
        
        # Process distance
        def parse_distance(dist):
            if isinstance(dist, str) and 'Other Location' in dist:
                return 300.0
            try:
                return float(dist) if pd.notna(dist) else 300.0
            except:
                return 300.0
        
        df_processed['distance'] = df_processed['Start DiffIn Meters'].apply(parse_distance)
        
        # Outstation flag
        remarks_columns = ['Attendance Reason Start', 'Attendance Reason End']
        df_processed['all_remarks'] = df_processed[remarks_columns].fillna('').apply(
            lambda x: ' '.join(x).lower(), axis=1
        )
        
        outstation_keywords = ['outstation', 'official event', 'official', 'event', 'market']
        df_processed['outstation_flag'] = df_processed['all_remarks'].apply(
            lambda x: 1 if any(keyword in x for keyword in outstation_keywords) else 0
        )
        
        # Apply business rules for attendance status
        conditions = [
            (df_processed['minutes_late'] <= 0) & (df_processed['distance'] <= 200),
            (df_processed['minutes_late'] <= 0) & (df_processed['distance'] > 200) & (df_processed['outstation_flag'] == 1),
            (df_processed['minutes_late'] > 0) & (df_processed['distance'] > 200) & (df_processed['outstation_flag'] == 0)
        ]
        choices = ['Present', 'Present', 'Absent']
        df_processed['attendance_status'] = np.select(conditions, choices, default='Present')
        
        # Additional features for analytics
        df_processed['day_name'] = df_processed['Attendance Date'].dt.day_name()
        df_processed['week_number'] = df_processed['Attendance Date'].dt.isocalendar().week
        df_processed['month'] = df_processed['Attendance Date'].dt.month_name()
        df_processed['year'] = df_processed['Attendance Date'].dt.year
        df_processed['is_late'] = (df_processed['minutes_late'] > 0).astype(int)
        df_processed['late_category'] = pd.cut(df_processed['minutes_late'], 
                                             bins=[-float('inf'), 0, 15, 30, 60, float('inf')],
                                             labels=['Early', '0-15 min', '15-30 min', '30-60 min', '60+ min'])
        
        return df_processed
    
    def get_overall_analytics(self):
        """Generate overall analytics"""
        if self.processed_data is None:
            return None
        
        df = self.processed_data
        
        analytics = {
            'total_records': len(df),
            'total_employees': df['User Name'].nunique(),
            'date_range': {
                'start': df['Attendance Date'].min(),
                'end': df['Attendance Date'].max(),
                'days': (df['Attendance Date'].max() - df['Attendance Date'].min()).days + 1
            },
            'attendance_stats': {
                'present': (df['attendance_status'] == 'Present').sum(),
                'absent': (df['attendance_status'] == 'Absent').sum(),
                'present_rate': ((df['attendance_status'] == 'Present').sum() / len(df)) * 100
            },
            'late_stats': {
                'total_late': (df['minutes_late'] > 0).sum(),
                'late_rate': ((df['minutes_late'] > 0).sum() / len(df)) * 100,
                'avg_late_minutes': df[df['minutes_late'] > 0]['minutes_late'].mean(),
                'max_late': df['minutes_late'].max()
            },
            'outstation_stats': {
                'outstation_count': df['outstation_flag'].sum(),
                'outstation_rate': (df['outstation_flag'].sum() / len(df)) * 100
            }
        }
        
        return analytics
    
    def apply_business_rules(self, minutes_late, distance, outstation_flag):
        """Apply business rules to determine attendance status"""
        # Present Condition 1: Check-in ≤ 09:15 AND distance ≤ 200m
        if minutes_late <= 0 and distance <= 200:
            return "Present"
        
        # Present Condition 2: Check-in ≤ 09:15 AND distance > 200m AND outstation
        elif minutes_late <= 0 and distance > 200 and outstation_flag == 1:
            return "Present"
        
        # Absent Condition 1: Check-in > 09:15 AND distance > 200m
        elif minutes_late > 0 and distance > 200 and outstation_flag == 0:
            return "Absent"
        
        # Else: Present (as per your rules)
        else:
            return "Absent"
    
    def predict_with_ml(self, features_df):
        """Predict attendance using ML model"""
        if not self.ml_model_loaded or self.model is None:
            return None, "ML model not loaded"
        
        try:
            # Ensure all required features are present
            for col in self.feature_columns:
                if col not in features_df.columns:
                    features_df[col] = 0
            
            # Select only the required features in correct order
            features_df = features_df[self.feature_columns]
            
            # Scale features if scaler is available
            if hasattr(self.scaler, 'transform'):
                features_scaled = self.scaler.transform(features_df)
            else:
                features_scaled = features_df
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            prediction_label = self.label_encoder.inverse_transform([prediction])[0]
            
            # Get prediction probabilities
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_scaled)[0]
                confidence = max(probabilities) * 100
            else:
                confidence = 100.0
            
            return prediction_label, f"{confidence:.1f}%"
            
        except Exception as e:
            return None, f"Prediction error: {str(e)}"
    
    def prepare_ml_features(self, minutes_late, distance, outstation_flag, day_of_week, is_weekend, checkin_hour):
        """Prepare features for ML prediction"""
        features = {
            'minutes_late': minutes_late,
            'distance': distance,
            'outstation_flag': outstation_flag,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'checkin_hour': checkin_hour
        }
        
        # Add distance categories
        distance_categories = ['Very Close', 'Close', 'Far', 'Very Far']
        distance_bins = [0, 50, 200, 500, float('inf')]
        
        try:
            distance_cat = pd.cut([distance], bins=distance_bins, labels=distance_categories)[0]
            for category in distance_categories:
                features[f'distance_{category}'] = 1 if category == distance_cat else 0
        except:
            for category in distance_categories:
                features[f'distance_{category}'] = 1 if category == 'Very Far' else 0
        
        return pd.DataFrame([features])
    
    def predict_single_attendance(self, attendance_date, start_time, distance_val, remarks):
        """Predict attendance for a single case using both rule-based and ML"""
        # Calculate features
        checkin_datetime = pd.to_datetime(f"{attendance_date} {start_time}")
        reference_time = pd.to_datetime(f"{attendance_date} 09:15:00")
        minutes_late = (checkin_datetime - reference_time).total_seconds() / 60
        
        # Process distance
        if isinstance(distance_val, str) and 'Other Location' in distance_val:
            distance = 300.0
        else:
            try:
                distance = float(distance_val)
            except:
                distance = 300.0
        
        # Outstation flag
        outstation_keywords = ['outstation', 'official event', 'official', 'event', 'market']
        outstation_flag = 1 if any(keyword in str(remarks).lower() for keyword in outstation_keywords) else 0
        
        # Rule-based prediction
        rule_based_pred = self.apply_business_rules(minutes_late, distance, outstation_flag)
        
        # ML prediction
        ml_pred = None
        ml_confidence = "N/A"
        
        if self.ml_model_loaded:
            day_of_week = checkin_datetime.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            checkin_hour = checkin_datetime.hour
            
            features_df = self.prepare_ml_features(minutes_late, distance, outstation_flag, 
                                                 day_of_week, is_weekend, checkin_hour)
            ml_pred, ml_confidence = self.predict_with_ml(features_df)
        
        return {
            'rule_based': rule_based_pred,
            'ml_prediction': ml_pred,
            'ml_confidence': ml_confidence,
            'minutes_late': minutes_late,
            'distance': distance,
            'outstation_flag': outstation_flag,
            'features': {
                'minutes_late': minutes_late,
                'distance': distance,
                'outstation_flag': outstation_flag,
                'day_of_week': day_of_week,
                'is_weekend': is_weekend,
                'checkin_hour': checkin_hour
            }
        }
    
    def create_ml_prediction_interface(self):
        """Create ML prediction interface"""
        st.markdown('<div class="section-header">🤖 Attendance Prediction</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📝 Prediction Input")
            
            # Create a form for better organization
            with st.form("prediction_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    attendance_date = st.date_input("Attendance Date", datetime.now())
                    start_time = st.time_input("Check-in Time", time(9, 0))
                
                with col2:
                    distance_val = st.text_input("Distance (meters or 'Other Location')", "100")
                    remarks = st.text_area("Remarks", "In the market")
                
                submitted = st.form_submit_button("🚀 Predict Attendance", use_container_width=True)
                
                if submitted:
                    with st.spinner("Making predictions..."):
                        result = self.predict_single_attendance(attendance_date, start_time, distance_val, remarks)
                        
                        # Display results in columns
                        st.markdown("---")
                        st.subheader("🎯 Prediction Results")
                        
                        res_col1, res_col2 = st.columns(2)
                        
                        with res_col1:
                            st.markdown('<div class="rule-based">', unsafe_allow_html=True)
                            st.subheader("📋 Rule-Based Prediction")
                            status_color = "🟢" if result['rule_based'] == 'Present' else "🔴"
                            st.markdown(f'<div style="font-size: 2.5rem; font-weight: bold; text-align: center;">{status_color} {result["rule_based"]}</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Business rules explanation
                            with st.expander("📖 Business Rules Applied"):
                                rules = [
                                    "✅ **Present if**: Check-in ≤ 09:15 AND distance ≤ 200m",
                                    "✅ **Present if**: Check-in ≤ 09:15 AND distance > 200m AND outstation",
                                    "❌ **Absent if**: Check-in > 09:15 AND distance > 200m",
                                    "❌ **Absent ** for all other cases"
                                ]
                                for rule in rules:
                                    st.write(rule)
                        
                        with res_col2:
                            if result['ml_prediction']:
                                st.markdown('<div class="ml-prediction">', unsafe_allow_html=True)
                                st.subheader("🤖 ML Prediction")
                                status_color = "🟢" if result['ml_prediction'] == 'Present' else "🔴"
                                st.markdown(f'<div style="font-size: 2.5rem; font-weight: bold; text-align: center;">{status_color} {result["ml_prediction"]}</div>', unsafe_allow_html=True)
                                st.markdown(f'<div style="font-size: 1.2rem; text-align: center;">Confidence: {result["ml_confidence"]}</div>', unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Feature details
                                with st.expander("🔍 ML Feature Details"):
                                    st.write(f"- **Minutes Late**: {result['minutes_late']:.1f}")
                                    st.write(f"- **Distance**: {result['distance']}m")
                                    st.write(f"- **Outstation**: {'Yes' if result['outstation_flag'] else 'No'}")
                                    st.write(f"- **Day of Week**: {result['features']['day_of_week']}")
                                    st.write(f"- **Weekend**: {'Yes' if result['features']['is_weekend'] else 'No'}")
                                    st.write(f"- **Check-in Hour**: {result['features']['checkin_hour']}:00")
                            else:
                                st.info("ML model not available. Using rule-based prediction only.")
        
        with col2:
            st.subheader("🔧 Model Information")
            
            if self.ml_model_loaded:
                st.success("✅ ML Model Loaded Successfully")
                
                # Model details
                st.write("**Model Details:**")
                st.write(f"- **Model Type**: {type(self.model).__name__}")
                st.write(f"- **Feature Count**: {len(self.feature_columns)}")
                st.write(f"- **Status**: Active")
                
                # Feature importance if available
                if hasattr(self.model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'feature': self.feature_columns,
                        'importance': self.model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    st.write("**Top Features:**")
                    for i, row in feature_importance.head(5).iterrows():
                        st.write(f"- {row['feature']}: {row['importance']:.3f}")
            else:
                st.warning("❌ No ML Model Loaded")
                st.info("To use ML predictions, upload a trained model in the sidebar.")
                
                with st.expander("💡 How to get a trained model"):
                    st.write("""
                    1. **Train a model** using your historical attendance data
                    2. **Save the model** as a .joblib file with:
                       - Trained classifier
                       - Feature scaler
                       - Label encoder
                       - Feature columns list
                    3. **Upload it** using the sidebar option
                    """)
    
    def create_overview_dashboard(self):
        """Create comprehensive overview dashboard"""
        if self.processed_data is None:
            st.warning("Please load data first")
            return
        
        analytics = self.get_overall_analytics()
        df = self.processed_data
        
        st.markdown('<div class="section-header">📊 Overall Analytics Dashboard</div>', unsafe_allow_html=True)
        
        # Key Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Employees", analytics['total_employees'])
        with col2:
            st.metric("Total Records", analytics['total_records'])
        with col3:
            st.metric("Present Rate", f"{analytics['attendance_stats']['present_rate']:.1f}%")
        with col4:
            st.metric("Late Rate", f"{analytics['late_stats']['late_rate']:.1f}%")
        with col5:
            st.metric("ML Ready", "✅" if self.ml_model_loaded else "❌")
        
        # First row of charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Attendance distribution pie chart
            attendance_counts = df['attendance_status'].value_counts()
            fig_attendance = px.pie(
                values=attendance_counts.values,
                names=attendance_counts.index,
                title='Attendance Distribution',
                color=attendance_counts.index,
                color_discrete_map={'Present': '#2ecc71', 'Absent': '#e74c3c'}
            )
            fig_attendance.update_layout(height=400)
            st.plotly_chart(fig_attendance, use_container_width=True)
        
        with col2:
            # Late arrival distribution
            late_data = df[df['minutes_late'] > 0]
            if not late_data.empty:
                fig_late_dist = px.histogram(
                    late_data, 
                    x='minutes_late',
                    title='Distribution of Late Arrivals (Minutes)',
                    nbins=20,
                    color_discrete_sequence=['#f39c12']
                )
                fig_late_dist.update_layout(height=400)
                st.plotly_chart(fig_late_dist, use_container_width=True)
            else:
                st.info("No late arrivals in the data")
        
        # Second row of charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily attendance trend
            daily_attendance = df.groupby('Attendance Date')['attendance_status'].value_counts().unstack().fillna(0)
            fig_daily_trend = go.Figure()
            if 'Present' in daily_attendance.columns:
                fig_daily_trend.add_trace(go.Scatter(
                    x=daily_attendance.index, 
                    y=daily_attendance['Present'],
                    mode='lines+markers',
                    name='Present',
                    line=dict(color='#2ecc71', width=3)
                ))
            if 'Absent' in daily_attendance.columns:
                fig_daily_trend.add_trace(go.Scatter(
                    x=daily_attendance.index, 
                    y=daily_attendance['Absent'],
                    mode='lines+markers',
                    name='Absent',
                    line=dict(color='#e74c3c', width=3)
                ))
            fig_daily_trend.update_layout(
                title='Daily Attendance Trend', 
                xaxis_title='Date', 
                yaxis_title='Count',
                height=400
            )
            st.plotly_chart(fig_daily_trend, use_container_width=True)
        
        with col2:
            # Day-wise attendance pattern
            day_attendance = df.groupby('day_name')['attendance_status'].value_counts().unstack().fillna(0)
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_attendance = day_attendance.reindex(day_order)
            
            fig_day_pattern = go.Figure()
            if 'Present' in day_attendance.columns:
                fig_day_pattern.add_trace(go.Bar(
                    x=day_attendance.index,
                    y=day_attendance['Present'],
                    name='Present',
                    marker_color='#2ecc71'
                ))
            if 'Absent' in day_attendance.columns:
                fig_day_pattern.add_trace(go.Bar(
                    x=day_attendance.index,
                    y=day_attendance['Absent'],
                    name='Absent',
                    marker_color='#e74c3c'
                ))
            fig_day_pattern.update_layout(
                title='Attendance Pattern by Day of Week', 
                barmode='stack',
                height=400
            )
            st.plotly_chart(fig_day_pattern, use_container_width=True)
        
        # Third row of charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Outstation analysis
            outstation_analysis = df.groupby('outstation_flag')['attendance_status'].value_counts().unstack().fillna(0)
            fig_outstation = px.bar(
                outstation_analysis,
                title='Attendance by Outstation Status',
                labels={'value': 'Count', 'variable': 'Status'},
                color_discrete_map={'Present': '#2ecc71', 'Absent': '#e74c3c'}
            )
            fig_outstation.update_layout(height=400)
            st.plotly_chart(fig_outstation, use_container_width=True)
        
        with col2:
            # Late category distribution
            late_cat_dist = df['late_category'].value_counts()
            fig_late_cat = px.pie(
                values=late_cat_dist.values,
                names=late_cat_dist.index,
                title='Late Arrival Categories'
            )
            fig_late_cat.update_layout(height=400)
            st.plotly_chart(fig_late_cat, use_container_width=True)
    
    def create_employee_analytics(self, employee_name=None, employee_code=None):
        """Create detailed analytics for a specific employee"""
        if self.processed_data is None:
            st.warning("Please load data first")
            return
        
        df = self.processed_data
        
        # Filter for specific employee
        if employee_name:
            employee_data = df[df['User Name'] == employee_name]
            employee_id = employee_name
        elif employee_code:
            employee_data = df[df['User Code'] == employee_code]
            employee_id = employee_code
        else:
            st.warning("Please select an employee")
            return
        
        if employee_data.empty:
            st.error("Employee not found in the data")
            return
        
        st.markdown(f'<div class="section-header">👤 Employee Analytics: {employee_id}</div>', unsafe_allow_html=True)
        
        # Employee summary metrics
        total_days = len(employee_data)
        present_days = (employee_data['attendance_status'] == 'Present').sum()
        absent_days = (employee_data['attendance_status'] == 'Absent').sum()
        late_days = (employee_data['minutes_late'] > 0).sum()
        attendance_rate = (present_days / total_days) * 100
        late_rate = (late_days / total_days) * 100
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Days", total_days)
        with col2:
            st.metric("Present Days", present_days)
        with col3:
            st.metric("Absent Days", absent_days)
        with col4:
            st.metric("Attendance Rate", f"{attendance_rate:.1f}%")
        with col5:
            st.metric("Late Days", late_days)
        
        # Employee charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Attendance trend for employee
            emp_trend = employee_data.groupby('Attendance Date')['attendance_status'].value_counts().unstack().fillna(0)
            fig_emp_trend = go.Figure()
            if 'Present' in emp_trend.columns:
                fig_emp_trend.add_trace(go.Scatter(
                    x=emp_trend.index, y=emp_trend['Present'],
                    mode='lines+markers', name='Present', line=dict(color='#2ecc71', width=3)
                ))
            if 'Absent' in emp_trend.columns:
                fig_emp_trend.add_trace(go.Scatter(
                    x=emp_trend.index, y=emp_trend['Absent'],
                    mode='lines+markers', name='Absent', line=dict(color='#e74c3c', width=3)
                ))
            fig_emp_trend.update_layout(title=f'Attendance Trend - {employee_id}', height=400)
            st.plotly_chart(fig_emp_trend, use_container_width=True)
        
        with col2:
            # Late minutes distribution for employee
            late_employee_data = employee_data[employee_data['minutes_late'] > 0]
            if not late_employee_data.empty:
                fig_emp_late = px.histogram(
                    late_employee_data,
                    x='minutes_late',
                    title=f'Late Arrival Distribution - {employee_id}',
                    nbins=15,
                    color_discrete_sequence=['#f39c12']
                )
                fig_emp_late.update_layout(height=400)
                st.plotly_chart(fig_emp_late, use_container_width=True)
            else:
                st.info(f"{employee_id} has no late arrivals")
        
        # Additional employee analytics
        col1, col2 = st.columns(2)
        
        with col1:
            # Day-wise pattern
            day_pattern = employee_data.groupby('day_name')['attendance_status'].value_counts().unstack().fillna(0)
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_pattern = day_pattern.reindex(day_order)
            
            fig_day_emp = go.Figure()
            if 'Present' in day_pattern.columns:
                fig_day_emp.add_trace(go.Bar(x=day_pattern.index, y=day_pattern['Present'], name='Present', marker_color='#2ecc71'))
            if 'Absent' in day_pattern.columns:
                fig_day_emp.add_trace(go.Bar(x=day_pattern.index, y=day_pattern['Absent'], name='Absent', marker_color='#e74c3c'))
            fig_day_emp.update_layout(title=f'Day-wise Pattern - {employee_id}', barmode='stack', height=400)
            st.plotly_chart(fig_day_emp, use_container_width=True)
        
        with col2:
            # Check-in time distribution
            employee_data_copy = employee_data.copy()
            employee_data_copy['checkin_hour'] = pd.to_datetime(employee_data_copy['Start Day Time'].astype(str)).dt.hour
            checkin_dist = employee_data_copy['checkin_hour'].value_counts().sort_index()
            fig_checkin = px.bar(
                x=checkin_dist.index, y=checkin_dist.values,
                title=f'Check-in Time Distribution - {employee_id}',
                labels={'x': 'Hour of Day', 'y': 'Count'},
                color_discrete_sequence=['#3498db']
            )
            fig_checkin.update_layout(height=400)
            st.plotly_chart(fig_checkin, use_container_width=True)
        
        # Detailed statistics table
        st.subheader("📈 Detailed Statistics")
        
        # Calculate various statistics
        stats_data = {
            'Metric': [
                'Average Late Minutes', 'Maximum Late Minutes', 'Early Arrivals',
                'On-time Arrivals', 'Outstation Days', 'Most Frequent Check-in Time',
                'Best Attendance Day', 'Worst Attendance Day'
            ],
            'Value': [
                f"{employee_data['minutes_late'].mean():.1f} min",
                f"{employee_data['minutes_late'].max():.1f} min",
                f"{(employee_data['minutes_late'] < 0).sum()} days",
                f"{(employee_data['minutes_late'] <= 0).sum()} days",
                f"{employee_data['outstation_flag'].sum()} days",
                f"{employee_data_copy['checkin_hour'].mode().iloc[0] if not employee_data_copy['checkin_hour'].mode().empty else 'N/A'}:00",
                f"{employee_data.groupby('day_name')['attendance_status'].apply(lambda x: (x == 'Present').mean()).idxmax() if not employee_data.empty else 'N/A'}",
                f"{employee_data.groupby('day_name')['attendance_status'].apply(lambda x: (x == 'Absent').mean()).idxmax() if not employee_data.empty else 'N/A'}"
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
        
        # Recent attendance records
        st.subheader("📋 Recent Attendance Records")
        recent_records = employee_data[['Attendance Date', 'Start Day Time', 'minutes_late', 
                                      'distance', 'outstation_flag', 'attendance_status']].tail(10)
        st.dataframe(recent_records.sort_values('Attendance Date', ascending=False), use_container_width=True)

    def create_comparative_analytics(self):
            """Create comparative analytics between employees - Clean & Fixed Version"""
            if self.processed_data is None:
                st.warning("Please load data first")
                return

            df = self.processed_data

            st.markdown('<div class="section-header">📈 Comparative Analytics</div>', unsafe_allow_html=True)

            # ===============================
            # Employee-level statistics
            # ===============================
            st.subheader("🏆 Employee Performance Ranking")

            # Calculate attendance stats by employee
            employee_stats = df.groupby('User Name').agg({
                'attendance_status': lambda x: (x == 'Present').mean() * 100,  # Attendance %
                'minutes_late': 'mean',  # Avg late minutes
                'User Code': 'first',    # Unique code
            }).rename(columns={
                'attendance_status': 'attendance_rate',
                'minutes_late': 'avg_late_minutes',
                'User Code': 'user_code'
            })

            # Add count of records
            employee_stats['record_count'] = df.groupby('User Name').size()

            # Reset index
            employee_stats = employee_stats.reset_index()

            # ===============================
            # High-level summary
            # ===============================
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Employees", len(employee_stats))
            with col2:
                avg_attendance = employee_stats['attendance_rate'].mean()
                st.metric("Average Attendance Rate", f"{avg_attendance:.1f}%")
            with col3:
                avg_late = employee_stats['avg_late_minutes'].mean()
                st.metric("Average Late Minutes", f"{avg_late:.1f}")

            # ===============================
            # Top vs Bottom performers
            # ===============================
            col1, col2 = st.columns(2)

            with col1:
                # Top 10 by attendance
                top_attendees = employee_stats.nlargest(10, 'attendance_rate')
                fig_top_attend = px.bar(
                    top_attendees,
                    x='User Name',
                    y='attendance_rate',
                    title='Top 10 Employees by Attendance Rate',
                    color='attendance_rate',
                    color_continuous_scale='Viridis',
                    labels={'attendance_rate': 'Attendance Rate %'}
                )
                fig_top_attend.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_top_attend)

            with col2:
                # Bottom 10 by attendance
                bottom_attendees = employee_stats.nsmallest(10, 'attendance_rate')
                fig_bottom_attend = px.bar(
                    bottom_attendees,
                    x='User Name',
                    y='attendance_rate',
                    title='Bottom 10 Employees by Attendance Rate',
                    color='attendance_rate',
                    color_continuous_scale='Reds',
                    labels={'attendance_rate': 'Attendance Rate %'}
                )
                fig_bottom_attend.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_bottom_attend)

            # Separate Low Performer Table
            st.subheader("🔻 10 Lowest Performing Employees")
            low_performers = employee_stats.nsmallest(10, 'attendance_rate')
            st.dataframe(
                low_performers[['User Name', 'user_code', 'attendance_rate', 'avg_late_minutes', 'record_count']]
                .sort_values('attendance_rate', ascending=True)
                .round(2),
                use_container_width=True
            )

            # ===============================
            # Late minutes comparison
            # ===============================
            st.subheader("⏰ Late Minutes Comparison")
            col1, col2 = st.columns(2)

            with col1:
                punctual_employees = employee_stats.nsmallest(10, 'avg_late_minutes')
                fig_punctual = px.bar(
                    punctual_employees,
                    x='User Name',
                    y='avg_late_minutes',
                    title='Top 10 Most Punctual Employees',
                    color='avg_late_minutes',
                    color_continuous_scale='Greens',
                    labels={'avg_late_minutes': 'Avg Late Minutes'}
                )
                fig_punctual.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_punctual)

            with col2:
                late_employees = employee_stats.nlargest(10, 'avg_late_minutes')
                fig_late = px.bar(
                    late_employees,
                    x='User Name',
                    y='avg_late_minutes',
                    title='Top 10 Most Frequently Late Employees',
                    color='avg_late_minutes',
                    color_continuous_scale='Oranges',
                    labels={'avg_late_minutes': 'Avg Late Minutes'}
                )
                fig_late.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_late)

            # ===============================
            # Department-wise analysis
            # ===============================
            if 'User Designation' in df.columns:
                st.subheader("🏢 Department-wise Analysis")

                dept_stats = df.groupby('User Designation').agg({
                    'attendance_status': lambda x: (x == 'Present').mean() * 100,
                    'minutes_late': 'mean',
                    'User Name': 'nunique'
                }).rename(columns={
                    'attendance_status': 'attendance_rate',
                    'minutes_late': 'avg_late_minutes',
                    'User Name': 'employee_count'
                }).reset_index()

                col1, col2 = st.columns(2)

                with col1:
                    fig_dept_attend = px.bar(
                        dept_stats,
                        x='User Designation',
                        y='attendance_rate',
                        title='Attendance Rate by Department',
                        color='attendance_rate',
                        labels={'attendance_rate': 'Attendance Rate %'}
                    )
                    fig_dept_attend.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_dept_attend)

                with col2:
                    fig_dept_late = px.box(
                        df,
                        x='User Designation',
                        y='minutes_late',
                        title='Late Minutes Distribution by Department'
                    )
                    fig_dept_late.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_dept_late)

            # ===============================
            # Interactive comparison table
            # ===============================
            st.subheader("📋 Employee Comparison Table")

            col1, col2 = st.columns(2)
            with col1:
                min_attendance = st.slider("Minimum Attendance Rate %", 0, 100, 0)
            with col2:
                max_late = st.slider("Maximum Average Late Minutes", 0, 120, 120)

            filtered_stats = employee_stats[
                (employee_stats['attendance_rate'] >= min_attendance) &
                (employee_stats['avg_late_minutes'] <= max_late)
            ]

            st.dataframe(
                filtered_stats[['User Name', 'user_code', 'attendance_rate', 'avg_late_minutes', 'record_count']]
                .sort_values('attendance_rate', ascending=False)
                .round(2),
                use_container_width=True
            )


def main():
    # Initialize session state
    if 'analytics_system' not in st.session_state:
        st.session_state.analytics_system = AdvancedAttendanceAnalyticsSystem()
    
    # Main header
    st.markdown('<div class="main-header">🤖📊 Advanced Attendance Analytics & ML System</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ System Configuration")
    
    # MChnages karni hai isme 
    # st.sidebar.subheader("🤖 ML Model Settings")
    # ml_model_path = st.sidebar.text_input("ML Model Path", "./attendance_model_fixed.joblib")
    # if st.sidebar.button("Load ML Model", type="primary", use_container_width=True):
    #     with st.spinner("Loading ML model..."):
    #         if st.session_state.analytics_system.load_ml_model(ml_model_path):
    #             st.sidebar.success("✅ ML model loaded successfully!")
    #         else:
    #             st.sidebar.error("❌ Failed to load ML model")
    
    # Data loading section
    st.sidebar.subheader("📊 Data Management")
    uploaded_file = st.sidebar.file_uploader("Upload Attendance Data", type=['xlsx', 'csv'])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp_uploaded_file.xlsx", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.session_state.analytics_system.load_and_process_data("temp_uploaded_file.xlsx"):
            st.sidebar.success("✅ Data loaded successfully!")
            
            # Show data summary
            with st.sidebar.expander("📈 Data Summary"):
                if st.session_state.analytics_system.processed_data is not None:
                    df = st.session_state.analytics_system.processed_data
                    st.write(f"**Records**: {len(df)}")
                    st.write(f"**Employees**: {df['User Name'].nunique()}")
                    st.write(f"**Date Range**: {df['Attendance Date'].min().date()} to {df['Attendance Date'].max().date()}")
                    st.write(f"**Present Rate**: {(df['attendance_status'] == 'Present').mean()*100:.1f}%")
                    st.write(f"**ML Model**: {'✅ Loaded' if st.session_state.analytics_system.ml_model_loaded else '❌ Not Loaded'}")
        else:
            st.sidebar.error("❌ Error loading data")
    
    # System status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 System Status")
    
    status_col1, status_col2 = st.sidebar.columns(2)
    with status_col1:
        st.metric("Data Loaded", "✅" if st.session_state.analytics_system.processed_data is not None else "❌")
    # with status_col2:
    #     st.metric("ML Model", "✅" if st.session_state.analytics_system.ml_model_loaded else "❌")
    
    # Main tabs
    # tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    #     # "🤖 ML Prediction", 
    #     "📊 Overview", 
    #     "👤 Employee Analytics", 
    #     "📈 Comparative", 
    #     "📋 Data Explorer", 
    #     "ℹ️ About"
    # ])
    tab2, tab3, tab4, tab5, tab6 = st.tabs([
        # "🤖 ML Prediction", 
        "📊 Overview", 
        "👤 Employee Analytics", 
        "📈 Comparative", 
        "📋 Data Explorer", 
        "ℹ️ About"
    ])
    
    # with tab1:
    #     st.session_state.analytics_system.create_ml_prediction_interface()
    
    with tab2:
        if st.session_state.analytics_system.processed_data is not None:
            st.session_state.analytics_system.create_overview_dashboard()
        else:
            st.info("📁 Please upload data to view analytics dashboard")
    
    with tab3:
        if st.session_state.analytics_system.processed_data is not None:
            st.markdown('<div class="section-header">Employee Search & Analytics</div>', unsafe_allow_html=True)
            

            search_option = st.radio("Search by:", ["Name", "Code"])
            
            if search_option == "Name":
                employee_names = st.session_state.analytics_system.processed_data['User Name'].unique()
                selected_employee = st.selectbox("Select Employee:", sorted(employee_names))
                if st.button("Generate Analytics", type="primary", use_container_width=True):
                    st.session_state.analytics_system.create_employee_analytics(employee_name=selected_employee)
            else:
                employee_codes = st.session_state.analytics_system.processed_data['User Code'].unique()
                selected_code = st.selectbox("Select Employee Code:", sorted(employee_codes))
                if st.button("Generate Analytics", type="primary", use_container_width=True):
                    st.session_state.analytics_system.create_employee_analytics(employee_code=selected_code)
        
            
        else:
            st.info("📁 Please upload data to search employees")
    
    with tab4:
        if st.session_state.analytics_system.processed_data is not None:
            st.session_state.analytics_system.create_comparative_analytics()
        else:
            st.info("📁 Please upload data for comparative analytics")
    
    with tab5:
        if st.session_state.analytics_system.processed_data is not None:
            st.markdown('<div class="section-header">Raw Data Explorer</div>', unsafe_allow_html=True)
            
            df = st.session_state.analytics_system.processed_data
            
            # Data filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_date = df['Attendance Date'].min().date()
                max_date = df['Attendance Date'].max().date()
                date_range = st.date_input("Date Range", [min_date, max_date])
            
            with col2:
                status_filter = st.multiselect("Attendance Status", 
                                             options=['Present', 'Absent'],
                                             default=['Present', 'Absent'])
            
            with col3:
                employees_filter = st.multiselect("Employees",
                                                options=sorted(df['User Name'].unique()))
            
            # Filter data
            filtered_data = df.copy()
            
            if len(date_range) == 2:
                filtered_data = filtered_data[
                    (filtered_data['Attendance Date'] >= pd.to_datetime(date_range[0])) &
                    (filtered_data['Attendance Date'] <= pd.to_datetime(date_range[1]))
                ]
            
            if status_filter:
                filtered_data = filtered_data[filtered_data['attendance_status'].isin(status_filter)]
            
            if employees_filter:
                filtered_data = filtered_data[filtered_data['User Name'].isin(employees_filter)]
            
            st.write(f"Showing {len(filtered_data)} of {len(df)} records")
            st.dataframe(filtered_data, use_container_width=True, height=400)
            
            # Download filtered data
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data",
                data=csv,
                file_name="filtered_attendance_data.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("📁 Please upload data to explore")
    
    with tab6:
        st.markdown("""
        ## 🤖📊 Advanced Attendance Analytics & ML System
        
        ### 🚀 Features:
        
        - **🤖 ML Prediction**: AI-powered attendance prediction with confidence scores
        - **📊 Overview Dashboard**: Comprehensive analytics with multiple visualizations
        - **👤 Employee Analytics**: Individual employee performance tracking
        - **📈 Comparative Analytics**: Compare employees and departments
        - **📋 Data Explorer**: Interactive data exploration and filtering
        
        ### 🔧 Technical Capabilities:
        
        - **Dual Prediction System**: Rule-based + ML model predictions
        - **Real-time Analytics**: Live data processing and visualization
        - **Interactive Charts**: Plotly-based interactive visualizations
        - **Data Export**: Download filtered datasets
        - **Model Management**: Load and use pre-trained ML models
        
        ### 📋 Business Rules Applied:
        
        1. **✅ Present**: Check-in ≤ 09:15 AND distance ≤ 200m
        2. **✅ Present**: Check-in ≤ 09:15 AND distance > 200m AND outstation
        3. **❌ Absent**: Check-in > 09:15 AND distance > 200m
        4. **✅ Present**: All other cases
        
        ### 📊 Analytics Included:
        
        - Attendance distribution and trends
        - Late arrival patterns and categories
        - Day-wise and weekly analysis
        - Outstation tracking
        - Employee performance rankings
        - Department-wise comparisons
        - Interactive filtering and exploration
        
        ### 🎯 Usage Guide:
        
        1. **Upload Data**: Use the sidebar to upload your attendance data
        2. **Load ML Model**: Optional - for AI-powered predictions
        3. **Explore Tabs**: Navigate through different analytics sections
        4. **Make Predictions**: Use the ML Prediction tab for real-time analysis
        5. **Export Results**: Download filtered data for further analysis
        
        ---
        
        *Built with Streamlit, Plotly, and Scikit-learn*
        """)
        
        # System status
        st.subheader("🔧 System Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Data Loaded", "✅ Yes" if st.session_state.analytics_system.processed_data is not None else "❌ No")
        # with col2:
        #     st.metric("ML Model Ready", "✅ Yes" if st.session_state.analytics_system.ml_model_loaded else "❌ No")
        with col3:
            st.metric("System Version", "3.0.0")

if __name__ == "__main__":
    main()