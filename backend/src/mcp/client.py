"""
Gemini MCP Terminal Client
Natural language interface to analytics MCP server via Gemini API

Usage:
    python client/gemini_terminal_client.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import google.generativeai as genai
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import MCP tools
from src.mcp.server import (
    initialize_server,
    get_schema,
    get_metrics,
    run_analysis,
    generate_dashboard,
    summarize_insights,
    list_loaded_datasets
)

# Load environment
load_dotenv()

# Configure Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("❌ Error: GOOGLE_API_KEY not found in .env file")
    print("   Get your key from: https://aistudio.google.com/app/apikey")
    sys.exit(1)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")  # Read from .env with fallback

genai.configure(api_key=GOOGLE_API_KEY)


class GeminiMCPClient:
    """
    Gemini-powered MCP client for analytics
    Uses Gemini to understand user intent and call appropriate MCP tools
    """
    
    def __init__(self):
        """Initialize Gemini client (MCP server initialized separately)"""
        
        # Initialize Gemini with function calling
        # Use model from .env (GEMINI_MODEL)
        self.model = genai.GenerativeModel(
            GEMINI_MODEL,  # ← Now reads from .env!
            generation_config={
                "temperature": 0.3,  # Lower temperature for more deterministic tool calls
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )
        
        # Conversation history
        self.chat_history = []
        
        # Current context (remembers last file analyzed)
        self.current_file = None
        self.current_dataset_id = None
    
    async def initialize(self):
        """Initialize MCP server (must be called from async context)"""
        print("🔧 Initializing MCP server...")
        init_result = await initialize_server()
        
        if init_result.get("success"):
            print(f"✅ {init_result.get('message')}")
            features = init_result.get('features', {})
            print(f"   ML Forecasting: {'✅' if features.get('ml_forecasting') else '❌'}")
            print(f"   AI Insights: {'✅' if features.get('ai_insights') else '❌'}")
        else:
            print(f"❌ MCP server initialization failed: {init_result.get('error')}")
            sys.exit(1)
        
        print()
    
    def _get_system_prompt(self) -> str:
        """System prompt for Gemini explaining available MCP tools"""
        
        context = ""
        if self.current_file:
            context = f"\nCurrent context: User is working with file '{self.current_file}'"
        
        return f"""You are an AI analytics assistant with access to powerful data analysis tools via MCP (Model Context Protocol).

{context}

Available MCP Tools:

1. **get_schema(file_path: str)**
   - Auto-detects schema from CSV/Excel files
   - Returns: time column, metrics, dimensions, data types
   - Use this FIRST when user mentions a new file

2. **get_metrics(file_path: str)**
   - Gets detailed statistics for all metrics
   - Returns: min, max, mean, data quality info
   - Use when user asks "what metrics are available"

3. **run_analysis(file_path, analysis_type, metrics, ...)**
   - Runs analytics on the dataset
   - Analysis types: time_series, kpi, forecast, anomaly_detection, dimension_breakdown, seasonality
   - Returns: Complete analysis results
   - Use based on user's question (forecast → use "forecast", trends → use "time_series")

4. **summarize_insights(file_path, analysis_type, metrics, insight_type)**
   - Generates AI-powered natural language insights
   - insight_type: "executive" (brief) or "detailed" (comprehensive)
   - Use AFTER running analysis to explain results

5. **generate_dashboard(file_path, metrics, analysis_types)**
   - Creates HTML dashboard with visualizations
   - Use when user wants "dashboard" or "visualizations"

6. **list_loaded_datasets()**
   - Lists all currently loaded datasets
   - Use when user asks "what files are loaded"

Decision Logic:
- If user mentions a file path → call get_schema() first
- If user asks to "analyze" or "show trends" → call run_analysis() with analysis_type="time_series"
- If user asks to "forecast" or "predict" → call run_analysis() with analysis_type="forecast"
- If user asks to "find anomalies" → call run_analysis() with analysis_type="anomaly_detection"
- If user asks about "breakdown by product/region" → call run_analysis() with analysis_type="dimension_breakdown"
- After any analysis → call summarize_insights() to explain results
- If user asks for "dashboard" → call generate_dashboard()

Response Format:
When you decide to call a tool, respond with JSON:
{{
  "action": "call_tool",
  "tool": "tool_name",
  "parameters": {{...}},
  "explanation": "Brief explanation of what you're doing"
}}

After receiving tool results, explain them clearly to the user in natural language.

Important:
- Always use exact file paths provided by user
- Remember the current file context between questions
- For multi-step tasks (analyze + insights), call tools sequentially
- Be helpful and explain what you're doing
"""
    
    async def chat(self, user_message: str) -> str:
        """
        Process user message and execute MCP tool calls
        
        Args:
            user_message: User's natural language request
            
        Returns:
            AI response with analysis results
        """
        
        # Add user message to history
        self.chat_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Build full conversation context
        conversation = self._get_system_prompt() + "\n\n"
        
        # Add conversation history
        for msg in self.chat_history[-10:]:  # Keep last 10 messages for context
            conversation += f"{msg['role'].upper()}: {msg['content']}\n\n"
        
        conversation += "ASSISTANT: "
        
        try:
            # Get Gemini's response
            response = self.model.generate_content(conversation)
            ai_response = response.text.strip()
            
            # Check if Gemini wants to call a tool
            if self._is_tool_call(ai_response):
                # Execute tool and get results
                tool_results = await self._execute_tool_sequence(ai_response, user_message)
                
                # Let Gemini explain the results
                explanation_prompt = f"""
Previous conversation:
USER: {user_message}

Tool execution results:
{json.dumps(tool_results, indent=2, default=str)}

Now explain these results to the user in natural language. Be clear, concise, and highlight key findings.
If there are insights, present them as bullet points.
"""
                
                explanation = self.model.generate_content(explanation_prompt)
                final_response = explanation.text.strip()
            else:
                final_response = ai_response
            
            # Add to history
            self.chat_history.append({
                "role": "assistant",
                "content": final_response
            })
            
            return final_response
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            self.chat_history.append({
                "role": "assistant",
                "content": error_msg
            })
            return error_msg
    
    def _is_tool_call(self, response: str) -> bool:
        """Check if Gemini's response contains a tool call"""
        response_lower = response.lower()
        
        # Check for JSON structure or tool names
        return (
            '"action"' in response and '"call_tool"' in response
        ) or (
            '"tool"' in response and (
                'get_schema' in response_lower or
                'run_analysis' in response_lower or
                'summarize_insights' in response_lower or
                'generate_dashboard' in response_lower or
                'get_metrics' in response_lower
            )
        )
    
    async def _execute_tool_sequence(self, ai_response: str, original_question: str) -> Dict[str, Any]:
        """
        Parse and execute tool calls from Gemini's response
        Handles multi-step tool sequences (e.g., analyze → insights)
        """
        
        results = {
            "steps": [],
            "final_result": None
        }
        
        try:
            # Extract JSON from response
            json_start = ai_response.find('{')
            json_end = ai_response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                return {"error": "No valid tool call found"}
            
            tool_call = json.loads(ai_response[json_start:json_end])
            
            tool_name = tool_call.get("tool")
            parameters = tool_call.get("parameters", {})
            explanation = tool_call.get("explanation", "")
            
            print(f"\n🔧 {explanation}")
            print(f"   Tool: {tool_name}")
            print(f"   Parameters: {json.dumps(parameters, indent=6)}")
            
            # Execute primary tool
            step_result = await self._call_mcp_tool(tool_name, parameters)
            results["steps"].append({
                "tool": tool_name,
                "result": step_result
            })
            
            # Update current file context
            if "file_path" in parameters:
                self.current_file = parameters["file_path"]
            
            # Auto-follow-up: If analysis was successful, get insights
            if tool_name == "run_analysis" and step_result.get("success"):
                print(f"\n💡 Generating AI insights...")
                
                insights_params = {
                    "file_path": parameters.get("file_path"),
                    "analysis_type": parameters.get("analysis_type"),
                    "metrics": parameters.get("metrics"),
                    "insight_type": "detailed"
                }
                
                insights_result = await self._call_mcp_tool("summarize_insights", insights_params)
                results["steps"].append({
                    "tool": "summarize_insights",
                    "result": insights_result
                })
                
                results["final_result"] = {
                    "analysis": step_result,
                    "insights": insights_result
                }
            else:
                results["final_result"] = step_result
            
            return results
            
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse tool call: {str(e)}"}
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}
    
    async def _call_mcp_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call the actual MCP tool"""
        
        try:
            if tool_name == "get_schema":
                return await get_schema(**parameters)
            
            elif tool_name == "get_metrics":
                return await get_metrics(**parameters)
            
            elif tool_name == "run_analysis":
                return await run_analysis(**parameters)
            
            elif tool_name == "summarize_insights":
                return await summarize_insights(**parameters)
            
            elif tool_name == "generate_dashboard":
                return await generate_dashboard(**parameters)
            
            elif tool_name == "list_loaded_datasets":
                return await list_loaded_datasets()
            
            else:
                return {"error": f"Unknown tool: {tool_name}"}
                
        except Exception as e:
            return {"error": f"Tool '{tool_name}' failed: {str(e)}"}
    
    def reset(self):
        """Reset conversation history"""
        self.chat_history = []
        self.current_file = None
        print("🔄 Conversation reset\n")


async def main():
    """Interactive terminal chat loop"""
    
    print("=" * 70)
    print("  🤖 GEMINI MCP ANALYTICS CLIENT - Terminal Interface")
    print("=" * 70)
    print()
    print("✨ I'm your AI analytics assistant powered by Gemini")
    print()
    print("📝 Examples:")
    print("   • Analyze C:\\data\\sales.csv")
    print("   • Show me revenue trends")
    print("   • Forecast revenue for the next 30 days")
    print("   • Find anomalies in cost data")
    print("   • Generate a dashboard")
    print()
    print("⌨️  Commands:")
    print("   • 'reset' - Clear conversation history")
    print("   • 'quit' or 'exit' - Exit program")
    print()
    print("=" * 70)
    print()
    
    # Initialize client
    client = GeminiMCPClient()
    await client.initialize()  # Initialize MCP server asynchronously
    
    print("✅ Ready! Ask me anything about your data.\n")
    
    # Chat loop
    while True:
        try:
            # Get user input
            user_input = input("\n💬 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye! Happy analyzing!")
                break
            
            if user_input.lower() == 'reset':
                client.reset()
                continue
            
            # Process message
            print()
            response = await client.chat(user_input)
            
            print(f"\n🤖 Assistant:\n{response}\n")
            print("-" * 70)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Happy analyzing!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())