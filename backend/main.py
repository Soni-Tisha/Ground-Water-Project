import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dotenv
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from langchain_groq import ChatGroq



# LangChain and PandasAI imports
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from pandasai import Agent
from pandasai.llm import OpenAI as PandasOpenAI
from pandasai.connectors import PandasConnector
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
dotenv.load_dotenv()

# Page configuration
st.set_page_config(
    page_title="INGRES Groundwater Assistant", 
    page_icon="💧", 
    layout="wide"
)

class GroundwaterAssistant:
    """Simplified Groundwater Assistant using PandasAI Agent"""
    
    def __init__(self, csv_path: str):
        """Initialize the assistant with CSV data and PandasAI agent"""
        self.df = self.load_data(csv_path)
        self.agent = None              # ✅ Always initialize
        self.langchain_agent = None  
        self.setup_pandas_ai_agent()
        
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load and preprocess the CSV data"""
        if not os.path.exists(csv_path):
            st.error(f"CSV file not found: {csv_path}")
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
                st.warning("GROQ API key not found. Please set GROQ_API_KEY in environment")
                self.langchain_agent = None
                return
            
            llm_groq = ChatGroq(
                groq_api_key=groq_key,
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0,
            )

            # ✅ Add handle_parsing_errors=True to fix parsing issues
            self.langchain_agent = create_pandas_dataframe_agent(
                llm_groq,
                self.df,
                verbose=False,  # Change to True for debugging
                return_intermediate_steps=False,
                allow_dangerous_code=True,
                handle_parsing_errors=True  # ADD THIS LINE
            )

        except Exception as e:
            st.error(f"Error setting up AI agent: {str(e)}")
            self.langchain_agent = None

    def query_data(self, user_question: str) -> Dict[str, Any]:
        """Process user query using Groq LangChain agent"""
        if not self.langchain_agent:
            return {
                "response": "Groq AI agent not available. Please check your GROQ_API_KEY configuration.",
                "data": None,
                "chart": None
            }

        try:
            if self.is_greeting_query(user_question):
                return {
                    "response": self.handle_greeting_query(user_question),
                    "data": None,
                    "chart": None
                }
            if self.is_list_query(user_question):
                list_result = self.handle_list_query(user_question)
                if list_result:
                    return list_result
            # ADD: Check if it's an aggregate query and handle directly
            if self.is_aggregate_query(user_question):
                direct_result = self.handle_aggregate_query(user_question)
                if direct_result:
                    relevant_data = self.extract_relevant_data(user_question)
                    chart = self.create_visualization(user_question, relevant_data)
                    return {
                        "response": direct_result,
                        "data": relevant_data,
                        "chart": chart
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
            
            # ADD: Determine response type and format accordingly
            response_type = self.determine_response_type(user_question)
            if response_type == "visualization_primary":
                # For trend/comparison queries - minimal text response
                response = self.create_minimal_response(user_question, response)
            elif response_type == "text_with_viz":
                # Keep the full response but ensure it's clean
                pass
            else:
                # Text only - keep as is but clean
                pass
            
            # if any(word in user_question.lower() for word in ['which districts', 'districts have', 'list']):
            #     response = self.format_list_response(response)

            relevant_data = self.extract_relevant_data(user_question)
            chart = self.create_visualization(user_question, relevant_data)

            return {
                "response": str(response),
                "data": relevant_data,
                "chart": chart
            }

        except Exception as e:
            if "parsing error" in str(e).lower():
                return {
                    "response": "I had trouble understanding that question. Could you please rephrase it or ask something more specific about the groundwater data?",
                    "data": None,
                    "chart": None
                }
            else:
                return {
                    "response": f"Error processing query: {str(e)}",
                    "data": None,
                    "chart": None
                }
        
    def format_list_response(self, response: str) -> str:
        """Format list responses properly"""
        # If response contains array format like ['item1', 'item2']
        import re
        array_match = re.search(r'\[(.*?)\]', response)
        if array_match:
            # Extract items and clean them
            items_str = array_match.group(1)
            items = [item.strip().strip("'\"") for item in items_str.split(',')]
            items = [item for item in items if item]  # Remove empty items
            
            if items:
                formatted_items = '\n'.join([f"• {item}" for item in items])
                return f"The districts with critical groundwater stages are:\n\n{formatted_items}"
        
        return response
    
    def clean_response(self, response: str) -> str:
        """Clean up agent response to remove action/thought artifacts"""
        # Remove action/thought patterns
        import re
        
        # Remove Action: python_repl_ast patterns
        response = re.sub(r'Action:\s*python_repl_ast.*?(?=Thought:|Final Answer:|$)', '', response, flags=re.DOTALL)
        
        # Remove Thought: patterns that are not followed by useful content
        response = re.sub(r'Thought:.*?(?=Final Answer:|Thought:|$)', '', response, flags=re.DOTALL)
        
        # Extract only Final Answer if present
        final_answer_match = re.search(r'Final Answer:\s*(.*?)(?=Action:|Thought:|$)', response, re.DOTALL)
        if final_answer_match:
            response = final_answer_match.group(1).strip()
        
        # Clean up extra whitespace
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
        simple_greetings  = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 
                    'how are you', 'how you doing' 'what\'s up', 'greetings']
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
            
            # Handle critical groundwater stages
            if 'critical' in query_lower and 'districts' in query_lower:
                if 'Stage_of_Extraction' in self.df.columns and 'District' in self.df.columns:
                    critical_districts = self.df[
                        self.df['Stage_of_Extraction'].str.contains('Critical', case=False, na=False)
                    ]['District'].unique()
                    
                    if len(critical_districts) > 0:
                        district_list = '\n'.join([f"• {district}" for district in sorted(critical_districts)])
                        return {
                            "response": f"Districts with critical groundwater stages:\n\n{district_list}",
                            "data": self.df[self.df['Stage_of_Extraction'].str.contains('Critical', case=False, na=False)],
                            "chart": None
                        }
            
            # Handle safe groundwater stages
            elif 'safe' in query_lower and 'districts' in query_lower:
                if 'Stage_of_Extraction' in self.df.columns and 'District' in self.df.columns:
                    safe_districts = self.df[
                        self.df['Stage_of_Extraction'].str.contains('Safe', case=False, na=False)
                    ]['District'].unique()
                    
                    if len(safe_districts) > 0:
                        district_list = '\n'.join([f"• {district}" for district in sorted(safe_districts)])
                        return {
                            "response": f"Districts with safe groundwater stages:\n\n{district_list}",
                            "data": self.df[self.df['Stage_of_Extraction'].str.contains('Safe', case=False, na=False)],
                            "chart": None
                        }
            
            # Handle states with specific conditions
            elif 'states' in query_lower:
                if 'highest recharge' in query_lower:
                    if 'Recharge_mm' in self.df.columns:
                        top_states = self.df.groupby('State')['Recharge_mm'].mean().nlargest(10)
                        state_list = '\n'.join([f"• {state}: {value:.2f} mm" for state, value in top_states.items()])
                        return {
                            "response": f"States with highest average recharge:\n\n{state_list}",
                            "data": self.df[self.df['State'].isin(top_states.index)],
                            "chart": None
                        }
                
                elif 'lowest groundwater' in query_lower:
                    if 'Groundwater_Level_m' in self.df.columns:
                        low_states = self.df.groupby('State')['Groundwater_Level_m'].mean().nsmallest(10)
                        state_list = '\n'.join([f"• {state}: {value:.2f} m" for state, value in low_states.items()])
                        return {
                            "response": f"States with lowest average groundwater levels:\n\n{state_list}",
                            "data": self.df[self.df['State'].isin(low_states.index)],
                            "chart": None
                        }
            
            # Generic district listing
            elif 'districts' in query_lower and 'all' in query_lower:
                if 'District' in self.df.columns:
                    all_districts = sorted(self.df['District'].unique())
                    district_list = '\n'.join([f"• {district}" for district in all_districts[:50]])  # Limit to 50
                    more_text = f"\n\n... and {len(all_districts) - 50} more districts" if len(all_districts) > 50 else ""
                    return {
                        "response": f"All districts in the dataset:\n\n{district_list}{more_text}",
                        "data": self.df[['District', 'State']].drop_duplicates(),
                        "chart": None
                    }
            
            # Generic state listing
            elif 'states' in query_lower and 'all' in query_lower:
                if 'State' in self.df.columns:
                    all_states = sorted(self.df['State'].unique())
                    state_list = '\n'.join([f"• {state}" for state in all_states])
                    return {
                        "response": f"All states in the dataset:\n\n{state_list}",
                        "data": self.df[['State']].drop_duplicates(),
                        "chart": None
                    }
            
            return None
            
        except Exception:
            return None

    def handle_aggregate_query(self, query: str) -> Optional[str]:
        """Handle aggregate queries directly with pandas"""
        try:
            query_lower = query.lower()
            
            # Extract state/year from query
            states = self.df['State'].unique()
            mentioned_state = None
            for state in states:
                if state.lower() in query_lower:
                    mentioned_state = state
                    break
            
            # Extract year
            import re
            year_match = re.search(r'\b(19|20)\d{2}\b', query)
            mentioned_year = int(year_match.group()) if year_match else None
            
            # Filter data
            filtered_df = self.df.copy()
            if mentioned_state:
                filtered_df = filtered_df[filtered_df['State'] == mentioned_state]
            if mentioned_year:
                filtered_df = filtered_df[filtered_df['Year'] == mentioned_year]
            
            if filtered_df.empty:
                return f"No data found for the specified criteria."
            
            # Handle different aggregate types
            if 'total' in query_lower or 'sum' in query_lower:
                if 'recharge' in query_lower and 'Recharge_mm' in filtered_df.columns:
                    total_val = filtered_df['Recharge_mm'].sum()
                    location = f" in {mentioned_state}" if mentioned_state else ""
                    year_str = f" in {mentioned_year}" if mentioned_year else ""
                    return f"The total recharge{location}{year_str} is {total_val:.2f} mm."
                elif 'extraction' in query_lower and 'Extraction_mm' in filtered_df.columns:
                    total_val = filtered_df['Extraction_mm'].sum()
                    location = f" in {mentioned_state}" if mentioned_state else ""
                    year_str = f" in {mentioned_year}" if mentioned_year else ""
                    return f"The total extraction{location}{year_str} is {total_val:.2f} mm."
            
            elif 'average' in query_lower or 'mean' in query_lower:
                if 'groundwater' in query_lower and 'Groundwater_Level_m' in filtered_df.columns:
                    avg_val = filtered_df['Groundwater_Level_m'].mean()
                    location = f" in {mentioned_state}" if mentioned_state else ""
                    year_str = f" in {mentioned_year}" if mentioned_year else ""
                    return f"The average groundwater level{location}{year_str} is {avg_val:.2f} m."
            
            return None
            
        except Exception:
            return None

    def determine_response_type(self, query: str) -> str:
        """Determine what type of response is needed"""
        query_lower = query.lower()
        
        # Visualization primary (minimal text + chart)
        viz_primary_keywords = ['trend', 'over time', 'change over years', 'show me', 'plot', 'graph']
        if any(keyword in query_lower for keyword in viz_primary_keywords):
            return "visualization_primary"
        
        # Text with visualization
        text_viz_keywords = ['compare', 'comparison', 'between', 'vs', 'highest', 'lowest']
        if any(keyword in query_lower for keyword in text_viz_keywords):
            return "text_with_viz"
        
        # Text only
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
        
        elif 'compare' in query_lower:
            return "Here's the comparison between the requested locations:"
        
        return "Here's the visualization for your query:"

    def extract_relevant_data(self, query: str) -> Optional[pd.DataFrame]:
        """Extract relevant data subset based on query keywords"""
        query_lower = query.lower()
        
        try:
            # Extract mentioned states
            available_states = self.df['State'].unique()
            mentioned_states = []
            for state in available_states:
                if state.lower() in query_lower:
                    mentioned_states.append(state)
            
            # For comparison queries, ensure we get all mentioned states
            if any(word in query_lower for word in ['compare', 'between', 'vs']):
                if len(mentioned_states) >= 2:
                    filtered_df = self.df[self.df['State'].isin(mentioned_states)]
                    return filtered_df if not filtered_df.empty else None
                elif len(mentioned_states) == 1:
                    # Only one state found, return its data
                    filtered_df = self.df[self.df['State'] == mentioned_states[0]]
                    return filtered_df if not filtered_df.empty else None
            
            # Extract year if mentioned
            years = self.df['Year'].dropna().unique()
            mentioned_year = None
            for year in years:
                if str(year) in query:
                    mentioned_year = year
                    break
            
            # Filter data based on extracted entities
            filtered_df = self.df.copy()
            
            if mentioned_states:
                filtered_df = filtered_df[filtered_df['State'].isin(mentioned_states)]
            
            if mentioned_year:
                filtered_df = filtered_df[filtered_df['Year'] == mentioned_year]
            
            # Limit results to avoid overwhelming display
            if len(filtered_df) > 20:
                filtered_df = filtered_df.head(20)
                
            return filtered_df if not filtered_df.empty else None
            
        except Exception:
            return self.df.head(10)  # Return sample data as fallback

        
    def check_data_availability(self, query: str) -> bool:
        """Check if the query parameters have available data"""
        query_lower = query.lower()
        
        # Check for state mentions
        states = self.df['State'].dropna().unique()
        mentioned_states = []
        for state in states:
            if state.lower() in query_lower:
                mentioned_states.append(state)
        
        # If states are mentioned but none found in data, return False
        state_keywords = ['state', 'states', 'maharashtra', 'gujarat', 'punjab', 'tamil nadu', 'karnataka', 'rajasthan', 'haryana', 'uttar pradesh', 'west bengal', 'bihar', 'odisha', 'andhra pradesh', 'telangana', 'kerala', 'madhya pradesh', 'assam', 'jharkhand', 'chhattisgarh']
        
        # Check if query mentions any state-like keywords
        has_state_reference = any(keyword in query_lower for keyword in state_keywords)
        
        if has_state_reference and not mentioned_states:
            # Check if any of the actual state names in our data match
            actual_states_mentioned = any(state.lower() in query_lower for state in states)
            if not actual_states_mentioned:
                return False
        
        # Check for year mentions
        years = self.df['Year'].dropna().unique()
        mentioned_years = []
        for year in years:
            if str(year) in query:
                mentioned_years.append(year)
        
        return True
    
    # Add this method after check_data_availability method:
    def format_response(self, response: str, query: str) -> str:
        """Format AI response for better readability"""
        query_lower = query.lower()
        
        # Check if it's a "which districts" or list-type question
        if any(keyword in query_lower for keyword in ['which districts', 'list of', 'districts have', 'which states']):
            # Try to extract array-like content and format as list
            if '[' in response and ']' in response:
                import re
                # Extract array content
                array_match = re.search(r'\[(.*?)\]', response)
                if array_match:
                    items = array_match.group(1).replace("'", "").split()
                    # Clean up items
                    cleaned_items = [item.strip().strip("',") for item in items if item.strip().strip("',")]
                    if cleaned_items:
                        formatted_list = "\n".join([f"• {item}" for item in cleaned_items])
                        return f"The districts with critical groundwater stages are:\n\n{formatted_list}"
        
        # Check for comparison queries with no data
        comparison_keywords = ['compare', 'comparison', 'between', 'vs']
        if any(keyword in query_lower for keyword in comparison_keywords):
            if 'no data' in response.lower() or 'not available' in response.lower():
                return "No data is available for the mentioned query parameters."
        
        return response
    
    def create_visualization(self, query: str, data: Optional[pd.DataFrame]) -> Optional[go.Figure]:
        """Create appropriate visualization based on query and data"""
        if data is None or data.empty:
            return None
            
        query_lower = query.lower()
        
        try:
            # ALWAYS create charts for these patterns (no data size restrictions)
            if any(word in query_lower for word in ['trend', 'over time', 'change', 'years', 'over years', 'show me']):
                return self.create_trend_chart(data)
            elif any(word in query_lower for word in ['compare', 'comparison', 'vs', 'between']):
                return self.create_comparison_chart(data)
            elif any(word in query_lower for word in ['distribution', 'stage', 'category']):
                return self.create_distribution_chart(data)
            elif any(word in query_lower for word in ['highest', 'lowest', 'max', 'min', 'top', 'bottom']):
                return self.create_ranking_chart(data, query_lower)
            elif any(word in query_lower for word in ['which districts', 'which states', 'list of']):
                return None
            # ADD: For aggregate queries with location/time context
            elif any(word in query_lower for word in ['total', 'sum', 'average']) and any(word in query_lower for word in ['state', 'year', 'district']):
                return self.create_context_chart(data, query_lower)
            else:
                return None  # Don't create default charts unless specifically needed
                        
        except Exception as e:
            st.error(f"Chart creation error: {str(e)}")
            return None
    
    
        
    def create_trend_chart(self, data: pd.DataFrame) -> Optional[go.Figure]:
        """Create trend line chart"""
        if 'Year' not in data.columns:
            return None
            
        # Group by year and calculate mean for numeric columns
        numeric_cols = ['Groundwater_Level_m', 'Recharge_mm', 'Extraction_mm']
        available_cols = [col for col in numeric_cols if col in data.columns]
        
        if not available_cols:
            return None
        
        # If multiple states, show separate lines for each state
        if 'State' in data.columns and data['State'].nunique() > 1:
            trend_data = data.groupby(['Year', 'State'])[available_cols[0]].mean().reset_index()
            fig = px.line(
                trend_data, 
                x='Year', 
                y=available_cols[0],
                color='State',
                title=f'{available_cols[0].replace("_", " ").title()} Trend Over Time by State',
                markers=True
            )
        else:
            trend_data = data.groupby('Year')[available_cols[0]].mean().reset_index()
            fig = px.line(
                trend_data, 
                x='Year', 
                y=available_cols[0],
                title=f'{available_cols[0].replace("_", " ").title()} Trend Over Time',
                markers=True
            )
        
        fig.update_layout(height=400)
        return fig
    
    def create_comparison_chart(self, data: pd.DataFrame) -> Optional[go.Figure]:
        """Create comparison bar chart"""
        numeric_cols = ['Groundwater_Level_m', 'Recharge_mm', 'Extraction_mm']
        available_cols = [col for col in numeric_cols if col in data.columns]
        
        if not available_cols or 'State' not in data.columns:
            return None
        
        # REMOVE the state count check - always create chart if data exists
        comparison_data = data.groupby('State')[available_cols[0]].mean().reset_index()
        
        fig = px.bar(
            comparison_data,
            x='State',
            y=available_cols[0],
            title=f'{available_cols[0].replace("_", " ").title()} Comparison Between States',
            color='State',
            text=available_cols[0]
        )
        
        # ADD: Better text formatting
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig.update_layout(height=400, xaxis_tickangle=-45, showlegend=False)
        return fig
    
    
        
        
    def create_distribution_chart(self, data: pd.DataFrame) -> Optional[go.Figure]:
        """Create distribution pie chart"""
        if 'Stage_of_Extraction' not in data.columns:
            return None
            
        stage_counts = data['Stage_of_Extraction'].value_counts()
        
        fig = px.pie(
            values=stage_counts.values,
            names=stage_counts.index,
            title='Distribution of Extraction Stages'
        )
        fig.update_layout(height=400)
        return fig
    
    def create_ranking_chart(self, data: pd.DataFrame, query: str) -> Optional[go.Figure]:
        """Create ranking chart for highest/lowest queries"""
        numeric_cols = ['Groundwater_Level_m', 'Recharge_mm', 'Extraction_mm']
        available_cols = [col for col in numeric_cols if col in data.columns]
        
        if not available_cols or 'District' not in data.columns:
            return None
            
        # Determine sorting order
        ascending = 'lowest' in query or 'min' in query
        
        # Group by district and sort
        ranking_data = data.groupby('District')[available_cols[0]].mean().reset_index()
        ranking_data = ranking_data.sort_values(available_cols[0], ascending=ascending).head(10)
        
        fig = px.bar(
            ranking_data,
            x='District',
            y=available_cols[0],
            title=f'{"Lowest" if ascending else "Highest"} {available_cols[0].replace("_", " ").title()} by District',
            color=available_cols[0],
            color_continuous_scale='RdYlBu' if ascending else 'Blues'
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        return fig
    
    def create_default_chart(self, data: pd.DataFrame) -> Optional[go.Figure]:
        """Create default visualization"""
        numeric_cols = ['Groundwater_Level_m', 'Recharge_mm', 'Extraction_mm']
        available_cols = [col for col in numeric_cols if col in data.columns and not data[col].isna().all()]
        
        if not available_cols:
            return None
            
        # Create histogram of the first available numeric column
        fig = px.histogram(
            data,
            x=available_cols[0],
            title=f'Distribution of {available_cols[0].replace("_", " ").title()}',
            nbins=20
        )
        fig.update_layout(height=400)
        return fig
    
    def create_context_chart(self, data: pd.DataFrame, query: str) -> Optional[go.Figure]:
        """Create contextual chart for aggregate queries"""
        numeric_cols = ['Groundwater_Level_m', 'Recharge_mm', 'Extraction_mm']
        available_cols = [col for col in numeric_cols if col in data.columns]
        
        if not available_cols:
            return None
        
        # Determine which metric to show based on query
        metric_col = available_cols[0]  # default
        if 'recharge' in query:
            metric_col = 'Recharge_mm' if 'Recharge_mm' in available_cols else available_cols[0]
        elif 'extraction' in query:
            metric_col = 'Extraction_mm' if 'Extraction_mm' in available_cols else available_cols[0]
        elif 'groundwater' in query or 'level' in query:
            metric_col = 'Groundwater_Level_m' if 'Groundwater_Level_m' in available_cols else available_cols[0]
        
        # Create appropriate grouping
        if 'State' in data.columns and data['State'].nunique() > 1:
            # Group by state
            grouped_data = data.groupby('State')[metric_col].sum().reset_index()
            fig = px.bar(
                grouped_data,
                x='State',
                y=metric_col,
                title=f'{metric_col.replace("_", " ").title()} by State',
                color='State'
            )
        elif 'Year' in data.columns and data['Year'].nunique() > 1:
            # Group by year
            grouped_data = data.groupby('Year')[metric_col].sum().reset_index()
            fig = px.bar(
                grouped_data,
                x='Year',
                y=metric_col,
                title=f'{metric_col.replace("_", " ").title()} by Year',
                color=metric_col
            )
        else:
            return None
        
        fig.update_layout(height=400, showlegend=False)
        return fig

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the dataset"""
        return {
            "total_records": len(self.df),
            "states_count": self.df['State'].nunique(),
            "years_range": (int(self.df['Year'].min()), int(self.df['Year'].max())),
            "available_years": sorted(self.df['Year'].unique().tolist()),
            "available_states": sorted(self.df['State'].unique().tolist())
        }

# Initialize the application
@st.cache_resource
def load_assistant():
    """Load and cache the groundwater assistant"""
    csv_path = "data/ground_water_data.csv"  # Adjust path as needed
    return GroundwaterAssistant(csv_path)

def main():
    """Main Streamlit application"""
    st.title("💧 INGRES Groundwater Data Assistant")
    st.markdown("*AI-powered assistant using LangChain and PandasAI for groundwater data analysis*")
    
    # Load assistant
    assistant = load_assistant()
    
    if assistant.df.empty:
        st.error("No data available. Please check the CSV file path.")
        return
    
    # Get data summary
    summary = assistant.get_data_summary()
    
    # Sidebar
    # with st.sidebar:
    #     st.header("📊 Data Overview")
    #     st.metric("Total Records", f"{summary['total_records']:,}")
    #     st.metric("States", summary['states_count'])
    #     st.metric("Years", f"{summary['years_range'][0]} - {summary['years_range'][1]}")
        
    #     st.header("🗺️ Available States")
    #     for state in summary['available_states'][:10]:
    #         st.write(f"• {state}")
    #     if len(summary['available_states']) > 10:
    #         st.write(f"... and {len(summary['available_states']) - 10} more")
    
    # Main content
    # col2 = st.columns([2, 1])
    

    
    # with col2:
    #     st.subheader("📈 Quick Stats")
    #     if 'Groundwater_Level_m' in assistant.df.columns:
    #         avg_level = assistant.df['Groundwater_Level_m'].mean()
    #         if not pd.isna(avg_level):
    #             st.metric("Avg Groundwater Level", f"{avg_level:.2f} m")
        
    #     if 'Stage_of_Extraction' in assistant.df.columns:
    #         most_common_stage = assistant.df['Stage_of_Extraction'].mode().iloc[0] if not assistant.df['Stage_of_Extraction'].mode().empty else "N/A"
    #         st.metric("Most Common Stage", most_common_stage)
    
    # # Chat Interface
    # st.subheader("🤖 Ask Questions About the Data")
    
    # # Example queries
    # with st.expander("💡 Example Queries", expanded=True):
    #     examples = [
    #         "What is the average groundwater level in Gujarat?",
    #         "Which state has the highest recharge in 2020?",
    #         "Show me the trend of groundwater extraction over years",
    #         "Compare groundwater levels between Maharashtra and Punjab",
    #         "Which districts have critical groundwater stages?",
    #         "What is the total recharge in Tamil Nadu in 2019?"
    #     ]
        
    #     for i, example in enumerate(examples, 1):
    #         st.write(f"{i}. {example}")
    
    # Query input
    user_query = st.text_area(
        "Enter your question:",
        placeholder="e.g., What is the average groundwater level in Gujarat in 2020?",
        height=100
    )
    
    # Process query
    if st.button("🔍 Ask Question", type="primary"):
        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Processing your question..."):
                result = assistant.query_data(user_query)
                
                # Display response
                st.subheader("🤖 AI Response")
                # st.write(result["response"])
                
                if result["chart"] and any(word in user_query.lower() for word in ['trend', 'show me', 'over time']):
                    # For visualization-primary queries, show minimal text
                    st.write(result["response"])
                    st.subheader("📊 Visualization")
                    st.plotly_chart(result["chart"], use_container_width=True)
                else:
                    # For other queries, show full response
                    st.write(result["response"])
                    
                    # Display chart if available
                    if result["chart"]:
                        st.subheader("📊 Visualization")
                        st.plotly_chart(result["chart"], use_container_width=True)
                # # Display chart if available
                # if result["chart"]:
                #     st.subheader("📊 Visualization")
                #     st.plotly_chart(result["chart"], use_container_width=True)
                
                # # Display data if available
                # if result["data"] is not None and not result["data"].empty:
                #     st.subheader("📋 Related Data")
                    
                #     # Download button
                #     csv = result["data"].to_csv(index=False)
                #     st.download_button(
                #         label="📥 Download Data",
                #         data=csv,
                #         file_name=f"groundwater_query_result.csv",
                #         mime="text/csv"
                #     )
                    
                #     st.dataframe(result["data"], use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    <strong>INGRES - India Ground Water Resource Estimation System</strong><br>
    Powered by LangChain & PandasAI
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()