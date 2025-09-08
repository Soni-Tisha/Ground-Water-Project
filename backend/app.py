from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils
import dotenv
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import json
import uvicorn

# LangChain imports
from langchain_experimental.agents import create_pandas_dataframe_agent
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
dotenv.load_dotenv()

app = FastAPI(
    title="INGRES Groundwater API",
    description="AI-powered groundwater data analysis API",
    version="1.0.0"
)

# Pydantic models
class QueryRequest(BaseModel):
    question: str

class DataSummary(BaseModel):
    total_records: int
    states_count: int
    years_range: tuple
    available_years: List[int]
    available_states: List[str]

class QueryResponse(BaseModel):
    response: str
    data: Optional[List[Dict]] = None
    chart: Optional[Dict] = None
    chart_type: Optional[str] = None

class GroundwaterAssistant:
    """Groundwater Assistant using PandasAI Agent"""
    
    def __init__(self, csv_path: str):
        """Initialize the assistant with CSV data and PandasAI agent"""
        self.df = self.load_data(csv_path)
        self.agent = None
        self.langchain_agent = None
        self.setup_pandas_ai_agent()
        
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load and preprocess the CSV data"""
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            return pd.DataFrame()
            
        df = pd.read_csv(csv_path)
        
        # Basic data cleaning
        df = df.dropna(subset=['State', 'Year'])
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        
        # Standardize column names
        numeric_columns = ['Groundwater_Level_m', 'Recharge_mm', 'Extraction_mm']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df
    
    def setup_pandas_ai_agent(self):
        """Setup PandasAI agent with Groq"""
        try:
            load_dotenv()
            groq_key = os.getenv("GROQ_API_KEY")
            
            if not groq_key or not groq_key.startswith("gsk_"):
                print("GROQ API key not found. Please set GROQ_API_KEY in environment")
                self.langchain_agent = None
                return
            
            llm_groq = ChatGroq(
                groq_api_key=groq_key,
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0,
            )

            self.langchain_agent = create_pandas_dataframe_agent(
                llm_groq,
                self.df,
                verbose=False,
                return_intermediate_steps=False,
                allow_dangerous_code=True,
                handle_parsing_errors=True
            )

        except Exception as e:
            print(f"Error setting up AI agent: {str(e)}")
            self.langchain_agent = None

    def query_data(self, user_question: str) -> Dict[str, Any]:
        """Process user query using Groq LangChain agent"""
        if not self.langchain_agent:
            return {
                "response": "Groq AI agent not available. Please check your GROQ_API_KEY configuration.",
                "data": None,
                "chart": None,
                "chart_type": None
            }

        try:
            if self.is_greeting_query(user_question):
                return {
                    "response": self.handle_greeting_query(user_question),
                    "data": None,
                    "chart": None,
                    "chart_type": None
                }
            
            if self.is_list_query(user_question):
                list_result = self.handle_list_query(user_question)
                if list_result:
                    return list_result
            
            if self.is_aggregate_query(user_question):
                direct_result = self.handle_aggregate_query(user_question)
                if direct_result:
                    relevant_data = self.extract_relevant_data(user_question)
                    chart_data, chart_type = self.create_visualization_json(user_question, relevant_data)
                    return {
                        "response": direct_result,
                        "data": relevant_data.to_dict('records') if relevant_data is not None else None,
                        "chart": chart_data,
                        "chart_type": chart_type
                    }
            
            # Enhanced prompt for better responses
            enhanced_question = f"""
            {user_question}
            
            Instructions:
            - For aggregate queries (sum, total, average), calculate the exact value
            - For trend/comparison questions, provide brief summary only
            - Keep responses concise and factual
            - Use actual data values, not estimates
            """

            response = self.langchain_agent.run(enhanced_question)
            response = self.clean_response(response)
            
            response_type = self.determine_response_type(user_question)
            if response_type == "visualization_primary":
                response = self.create_minimal_response(user_question, response)

            relevant_data = self.extract_relevant_data(user_question)
            chart_data, chart_type = self.create_visualization_json(user_question, relevant_data)

            return {
                "response": str(response),
                "data": relevant_data.to_dict('records') if relevant_data is not None else None,
                "chart": chart_data,
                "chart_type": chart_type
            }

        except Exception as e:
            if "parsing error" in str(e).lower():
                return {
                    "response": "I had trouble understanding that question. Could you please rephrase it or ask something more specific about the groundwater data?",
                    "data": None,
                    "chart": None,
                    "chart_type": None
                }
            else:
                return {
                    "response": f"Error processing query: {str(e)}",
                    "data": None,
                    "chart": None,
                    "chart_type": None
                }

    def create_visualization_json(self, query: str, data: Optional[pd.DataFrame]) -> tuple[Optional[Dict], Optional[str]]:
        """Create visualization data in JSON format for frontend"""
        if data is None or data.empty:
            return None, None
            
        query_lower = query.lower()
        
        try:
            if any(word in query_lower for word in ['trend', 'over time', 'change', 'years', 'over years', 'show me']):
                return self.create_trend_chart_json(data)
            elif any(word in query_lower for word in ['compare', 'comparison', 'vs', 'between']):
                return self.create_comparison_chart_json(data)
            elif any(word in query_lower for word in ['distribution', 'stage', 'category']):
                return self.create_distribution_chart_json(data)
            elif any(word in query_lower for word in ['highest', 'lowest', 'max', 'min', 'top', 'bottom']):
                return self.create_ranking_chart_json(data, query_lower)
            elif any(word in query_lower for word in ['total', 'sum', 'average']) and any(word in query_lower for word in ['state', 'year', 'district']):
                return self.create_context_chart_json(data, query_lower)
            else:
                return None, None
                        
        except Exception as e:
            print(f"Chart creation error: {str(e)}")
            return None, None

    def create_trend_chart_json(self, data: pd.DataFrame) -> tuple[Dict, str]:
        """Create trend line chart data in JSON format"""
        if 'Year' not in data.columns:
            return None, None
            
        numeric_cols = ['Groundwater_Level_m', 'Recharge_mm', 'Extraction_mm']
        available_cols = [col for col in numeric_cols if col in data.columns]
        
        if not available_cols:
            return None, None
        
        chart_data = {
            "type": "line",
            "title": f'{available_cols[0].replace("_", " ").title()} Trend Over Time',
            "xAxis": {
                "label": "Year",
                "type": "category"
            },
            "yAxis": {
                "label": available_cols[0].replace("_", " ").title(),
                "type": "value"
            },
            "series": []
        }
        
        if 'State' in data.columns and data['State'].nunique() > 1:
            # Multiple states
            for state in data['State'].unique():
                state_data = data[data['State'] == state]
                trend_data = state_data.groupby('Year')[available_cols[0]].mean().reset_index()
                
                chart_data["series"].append({
                    "name": state,
                    "type": "line",
                    "data": [
                        {"x": row['Year'], "y": row[available_cols[0]]} 
                        for _, row in trend_data.iterrows()
                    ]
                })
            chart_data["title"] += " by State"
        else:
            # Single series
            trend_data = data.groupby('Year')[available_cols[0]].mean().reset_index()
            chart_data["series"].append({
                "name": available_cols[0].replace("_", " ").title(),
                "type": "line",
                "data": [
                    {"x": row['Year'], "y": row[available_cols[0]]} 
                    for _, row in trend_data.iterrows()
                ]
            })
        
        return chart_data, "line"

    def create_comparison_chart_json(self, data: pd.DataFrame) -> tuple[Dict, str]:
        """Create comparison bar chart data in JSON format"""
        numeric_cols = ['Groundwater_Level_m', 'Recharge_mm', 'Extraction_mm']
        available_cols = [col for col in numeric_cols if col in data.columns]
        
        if not available_cols or 'State' not in data.columns:
            return None, None
        
        comparison_data = data.groupby('State')[available_cols[0]].mean().reset_index()
        
        chart_data = {
            "type": "bar",
            "title": f'{available_cols[0].replace("_", " ").title()} Comparison Between States',
            "xAxis": {
                "label": "State",
                "type": "category"
            },
            "yAxis": {
                "label": available_cols[0].replace("_", " ").title(),
                "type": "value"
            },
            "series": [{
                "name": available_cols[0].replace("_", " ").title(),
                "type": "bar",
                "data": [
                    {"x": row['State'], "y": row[available_cols[0]]} 
                    for _, row in comparison_data.iterrows()
                ]
            }]
        }
        
        return chart_data, "bar"

    def create_distribution_chart_json(self, data: pd.DataFrame) -> tuple[Dict, str]:
        """Create distribution pie chart data in JSON format"""
        if 'Stage_of_Extraction' not in data.columns:
            return None, None
            
        stage_counts = data['Stage_of_Extraction'].value_counts()
        
        chart_data = {
            "type": "pie",
            "title": "Distribution of Extraction Stages",
            "series": [{
                "name": "Distribution",
                "type": "pie",
                "data": [
                    {"name": stage, "value": count} 
                    for stage, count in stage_counts.items()
                ]
            }]
        }
        
        return chart_data, "pie"

    def create_ranking_chart_json(self, data: pd.DataFrame, query: str) -> tuple[Dict, str]:
        """Create ranking chart data in JSON format"""
        numeric_cols = ['Groundwater_Level_m', 'Recharge_mm', 'Extraction_mm']
        available_cols = [col for col in numeric_cols if col in data.columns]
        
        if not available_cols or 'District' not in data.columns:
            return None, None
            
        ascending = 'lowest' in query or 'min' in query
        
        ranking_data = data.groupby('District')[available_cols[0]].mean().reset_index()
        ranking_data = ranking_data.sort_values(available_cols[0], ascending=ascending).head(10)
        
        chart_data = {
            "type": "bar",
            "title": f'{"Lowest" if ascending else "Highest"} {available_cols[0].replace("_", " ").title()} by District',
            "xAxis": {
                "label": "District",
                "type": "category"
            },
            "yAxis": {
                "label": available_cols[0].replace("_", " ").title(),
                "type": "value"
            },
            "series": [{
                "name": available_cols[0].replace("_", " ").title(),
                "type": "bar",
                "data": [
                    {"x": row['District'], "y": row[available_cols[0]]} 
                    for _, row in ranking_data.iterrows()
                ]
            }]
        }
        
        return chart_data, "bar"

    def create_context_chart_json(self, data: pd.DataFrame, query: str) -> tuple[Dict, str]:
        """Create contextual chart data for aggregate queries in JSON format"""
        numeric_cols = ['Groundwater_Level_m', 'Recharge_mm', 'Extraction_mm']
        available_cols = [col for col in numeric_cols if col in data.columns]
        
        if not available_cols:
            return None, None
        
        metric_col = available_cols[0]
        if 'recharge' in query:
            metric_col = 'Recharge_mm' if 'Recharge_mm' in available_cols else available_cols[0]
        elif 'extraction' in query:
            metric_col = 'Extraction_mm' if 'Extraction_mm' in available_cols else available_cols[0]
        elif 'groundwater' in query or 'level' in query:
            metric_col = 'Groundwater_Level_m' if 'Groundwater_Level_m' in available_cols else available_cols[0]
        
        chart_data = {
            "type": "bar",
            "title": f'{metric_col.replace("_", " ").title()}',
            "xAxis": {
                "type": "category"
            },
            "yAxis": {
                "label": metric_col.replace("_", " ").title(),
                "type": "value"
            },
            "series": [{
                "name": metric_col.replace("_", " ").title(),
                "type": "bar",
                "data": []
            }]
        }
        
        if 'State' in data.columns and data['State'].nunique() > 1:
            chart_data["xAxis"]["label"] = "State"
            chart_data["title"] += " by State"
            grouped_data = data.groupby('State')[metric_col].sum().reset_index()
            chart_data["series"][0]["data"] = [
                {"x": row['State'], "y": row[metric_col]} 
                for _, row in grouped_data.iterrows()
            ]
        elif 'Year' in data.columns and data['Year'].nunique() > 1:
            chart_data["xAxis"]["label"] = "Year"
            chart_data["title"] += " by Year"
            grouped_data = data.groupby('Year')[metric_col].sum().reset_index()
            chart_data["series"][0]["data"] = [
                {"x": row['Year'], "y": row[metric_col]} 
                for _, row in grouped_data.iterrows()
            ]
        else:
            return None, None
        
        return chart_data, "bar"

    # Keep all other methods from the original class...
    def clean_response(self, response: str) -> str:
        """Clean up agent response to remove action/thought artifacts"""
        import re
        response = re.sub(r'Action:\s*python_repl_ast.*?(?=Thought:|Final Answer:|$)', '', response, flags=re.DOTALL)
        response = re.sub(r'Thought:.*?(?=Final Answer:|Thought:|$)', '', response, flags=re.DOTALL)
        
        final_answer_match = re.search(r'Final Answer:\s*(.*?)(?=Action:|Thought:|$)', response, re.DOTALL)
        if final_answer_match:
            response = final_answer_match.group(1).strip()
        
        response = re.sub(r'\n\s*\n', '\n\n', response)
        response = response.strip()
        return response
    
    def is_aggregate_query(self, query: str) -> bool:
        """Check if query is asking for aggregate calculations"""
        query_lower = query.lower()
        aggregate_keywords = ['total', 'sum', 'average', 'mean', 'maximum', 'minimum', 'count']
        return any(keyword in query_lower for keyword in aggregate_keywords)
    
    def is_greeting_query(self, query: str) -> bool:
        """Check if query is a simple greeting"""
        query_lower = query.lower().strip()
        simple_greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 
                           'how are you', 'how you doing', 'what\'s up', 'greetings']
        return query_lower in simple_greetings 
    
    def handle_greeting_query(self, query: str) -> str:
        """Handle greeting queries without using the agent"""
        query_lower = query.lower().strip()
        
        if 'how are you' in query_lower:
            return "I'm doing well, thanks for asking! I'm here to help with any questions about groundwater data."
        elif query_lower in ['hello', 'hi', 'hey']:
            return "Hello! I'm here to help you analyze groundwater data."
        elif query_lower in ['good morning', 'good afternoon', 'good evening']:
            return f"Good {query_lower.split()[-1]}! I'm here to assist you with groundwater data."
        else:
            return "Hello! I'm your groundwater data assistant. Feel free to ask me any questions about the data."
        
    def is_list_query(self, query: str) -> bool:
        """Check if query is asking for a list of items"""
        query_lower = query.lower()
        list_patterns = [
            'which districts', 'which states', 'list districts', 'list states',
            'what districts', 'what states', 'all districts', 'all states',
            'districts have', 'states have', 'districts with', 'states with'
        ]
        return any(pattern in query_lower for pattern in list_patterns)

    def handle_list_query(self, query: str) -> Dict[str, Any]:
        """Handle list queries directly with pandas"""
        try:
            query_lower = query.lower()
            
            if 'critical' in query_lower and 'districts' in query_lower:
                if 'Stage_of_Extraction' in self.df.columns and 'District' in self.df.columns:
                    critical_districts = self.df[
                        self.df['Stage_of_Extraction'].str.contains('Critical', case=False, na=False)
                    ]['District'].unique()
                    
                    if len(critical_districts) > 0:
                        district_list = '\n'.join([f"• {district}" for district in sorted(critical_districts)])
                        return {
                            "response": f"Districts with critical groundwater stages:\n\n{district_list}",
                            "data": self.df[self.df['Stage_of_Extraction'].str.contains('Critical', case=False, na=False)].to_dict('records'),
                            "chart": None,
                            "chart_type": None
                        }
            
            return None
        except Exception:
            return None

    def handle_aggregate_query(self, query: str) -> Optional[str]:
        """Handle aggregate queries directly with pandas"""
        try:
            query_lower = query.lower()
            
            states = self.df['State'].unique()
            mentioned_state = None
            for state in states:
                if state.lower() in query_lower:
                    mentioned_state = state
                    break
            
            import re
            year_match = re.search(r'\b(19|20)\d{2}\b', query)
            mentioned_year = int(year_match.group()) if year_match else None
            
            filtered_df = self.df.copy()
            if mentioned_state:
                filtered_df = filtered_df[filtered_df['State'] == mentioned_state]
            if mentioned_year:
                filtered_df = filtered_df[filtered_df['Year'] == mentioned_year]
            
            if filtered_df.empty:
                return f"No data found for the specified criteria."
            
            if 'total' in query_lower or 'sum' in query_lower:
                if 'recharge' in query_lower and 'Recharge_mm' in filtered_df.columns:
                    total_val = filtered_df['Recharge_mm'].sum()
                    location = f" in {mentioned_state}" if mentioned_state else ""
                    year_str = f" in {mentioned_year}" if mentioned_year else ""
                    return f"The total recharge{location}{year_str} is {total_val:.2f} mm."
            
            return None
        except Exception:
            return None

    def determine_response_type(self, query: str) -> str:
        """Determine what type of response is needed"""
        query_lower = query.lower()
        
        viz_primary_keywords = ['trend', 'over time', 'change over years', 'show me', 'plot', 'graph']
        if any(keyword in query_lower for keyword in viz_primary_keywords):
            return "visualization_primary"
        
        text_viz_keywords = ['compare', 'comparison', 'between', 'vs', 'highest', 'lowest']
        if any(keyword in query_lower for keyword in text_viz_keywords):
            return "text_with_viz"
        
        return "text_only"

    def create_minimal_response(self, query: str, original_response: str) -> str:
        """Create minimal response for visualization-primary queries"""
        query_lower = query.lower()
        
        if 'trend' in query_lower:
            if 'extraction' in query_lower:
                return "Here's the groundwater extraction trend over the years:"
            elif 'recharge' in query_lower:
                return "Here's the groundwater recharge trend over the years:"
            else:
                return "Here's the groundwater level trend over the years:"
        
        return "Here's the visualization for your query:"

    def extract_relevant_data(self, query: str) -> Optional[pd.DataFrame]:
        """Extract relevant data subset based on query keywords"""
        query_lower = query.lower()
        
        try:
            available_states = self.df['State'].unique()
            mentioned_states = []
            for state in available_states:
                if state.lower() in query_lower:
                    mentioned_states.append(state)
            
            years = self.df['Year'].dropna().unique()
            mentioned_year = None
            for year in years:
                if str(year) in query:
                    mentioned_year = year
                    break
            
            filtered_df = self.df.copy()
            
            if mentioned_states:
                filtered_df = filtered_df[filtered_df['State'].isin(mentioned_states)]
            
            if mentioned_year:
                filtered_df = filtered_df[filtered_df['Year'] == mentioned_year]
            
            if len(filtered_df) > 20:
                filtered_df = filtered_df.head(20)
                
            return filtered_df if not filtered_df.empty else None
            
        except Exception:
            return self.df.head(10)

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the dataset"""
        return {
            "total_records": len(self.df),
            "states_count": self.df['State'].nunique(),
            "years_range": (int(self.df['Year'].min()), int(self.df['Year'].max())),
            "available_years": sorted(self.df['Year'].unique().tolist()),
            "available_states": sorted(self.df['State'].unique().tolist())
        }

# Initialize assistant globally
assistant = None

@app.on_event("startup")
async def startup_event():
    """Initialize the assistant on startup"""
    global assistant
    csv_path = "data/ground_water_data.csv"  # Adjust path as needed
    assistant = GroundwaterAssistant(csv_path)
    print("Groundwater Assistant initialized")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "INGRES Groundwater Data Assistant API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "data_loaded": not assistant.df.empty if assistant else False}

@app.get("/summary", response_model=DataSummary)
async def get_data_summary():
    """Get data summary"""
    if not assistant or assistant.df.empty:
        raise HTTPException(status_code=500, detail="Data not available")
    
    summary = assistant.get_data_summary()
    return DataSummary(**summary)

@app.post("/query", response_model=QueryResponse)
async def query_data(request: QueryRequest):
    """Process user query"""
    if not assistant or assistant.df.empty:
        raise HTTPException(status_code=500, detail="Data not available")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        result = assistant.query_data(request.question)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/states")
async def get_states():
    """Get list of available states"""
    if not assistant or assistant.df.empty:
        raise HTTPException(status_code=500, detail="Data not available")
    
    return {"states": sorted(assistant.df['State'].unique().tolist())}

@app.get("/years")
async def get_years():
    """Get list of available years"""
    if not assistant or assistant.df.empty:
        raise HTTPException(status_code=500, detail="Data not available")
    
    return {"years": sorted(assistant.df['Year'].unique().tolist())}

@app.get("/columns")
async def get_columns():
    """Get list of available columns"""
    if not assistant or assistant.df.empty:
        raise HTTPException(status_code=500, detail="Data not available")
    
    return {"columns": assistant.df.columns.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)