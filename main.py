import streamlit as st
import pandas as pd
import plotly.express as px
import spacy
import subprocess
import importlib
import os
import google.generativeai as genai
import io # Added for plot download

# --- Spacy Model Download (Moved to a more robust location if not present) ---
# This part is generally better handled outside the main app flow or with st.cache_resource
# However, keeping it for initial setup as per original code.
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.warning("Spacy model 'en_core_web_sm' not found. Downloading...")
    try:
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        importlib.invalidate_caches()
        nlp = spacy.load("en_core_web_sm")
        st.success("Spacy model downloaded successfully!")
    except Exception as e:
        st.error(f"Failed to download Spacy model: {e}. Some functionalities might be limited.")


# Configure Google Gemini API Key
# 🔐 Secure: Best practice is to use st.secrets for Streamlit Cloud deployment
# For local, you can set it as an environment variable or put it directly here for testing (NOT RECOMMENDED FOR PRODUCTION)
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# Example using st.secrets:
try:
    genai.configure(api_key=st.secrets["AIzaSyANsbcUixBinJPxdnEzLggkviHdZAswFQA"])
except KeyError:
    st.error("Gemini API Key not found in Streamlit secrets. Please add it to .streamlit/secrets.toml or set as environment variable GEMINI_API_KEY.")
    st.stop() # Stop execution if API key is missing

st.set_page_config(page_title="QueryBuddy AI", layout="wide", initial_sidebar_state="expanded")
st.title("🤖 QueryBuddy AI - Talk to Your Data (Powered by Google Cloud)")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- About Section ---
with st.expander("👋 About QueryBuddy AI"):
    st.markdown("""
    Welcome to QueryBuddy AI! This application allows you to interact with your CSV or Excel datasets using natural language queries.
    Simply upload your file, and start asking questions about your data. QueryBuddy AI uses Google's Gemini Pro model
    to understand your queries and generate Python code (Pandas and Plotly) to analyze and visualize your data.

    **How to Use:**
    1.  **Upload your Data:** Use the file uploader to select a CSV or Excel file.
    2.  **Explore Data:** Review the data preview and profiling sections.
    3.  **Ask Questions:** Type your questions in the chat input or click on suggested questions.
    4.  **Review & Edit Code:** You can view and even edit the AI-generated Python code if you need to fine-tune the analysis.
    5.  **Download Plots:** Download generated plots as PNG images.

    **Disclaimer:** The AI model generates code. While efforts are made to make it safe, always review the generated code, especially for sensitive operations.
    """)

uploaded_file = st.file_uploader("📂 Upload your CSV or Excel file", type=["csv", "xlsx"])

# --- Helper function for safer exec() ---
def safe_exec(code, df):
    local_vars = {"df": df, "pd": pd, "px": px}
    # Limit built-ins for security. Only essential ones are allowed.
    safe_globals = {
        "__builtins__": {
            "True": True, "False": False, "None": None,
            "int": int, "float": float, "str": str, "bool": bool,
            "list": list, "dict": dict, "tuple": tuple, "set": set,
            "range": range, "len": len, "sum": sum, "min": min, "max": max,
            "abs": abs, "round": round, "print": print, # Print could be allowed for debugging, but be cautious
            "Exception": Exception, "TypeError": TypeError, "ValueError": ValueError, "KeyError": KeyError
        },
        "pd": pd,
        "px": px,
        "df": df,
    }
    exec(code, safe_globals, local_vars) # Use safe_globals for global context
    return local_vars.get("fig", None)

def gpt_generate_code(query, df_head):
    prompt = f"""
You are an expert Python data analyst. Your task is to convert a natural language question into executable Python code.
You must use the Pandas library for data manipulation and the Plotly Express library for data visualization.

Here's the user's question: "{query}"

Here's a preview of the DataFrame (df) structure:
{df_head.to_string(index=False)}

**Instructions:**
1.  **ALWAYS** start your code with `result = ...` for data manipulation, and `fig = ...` for plotting if a plot is requested.
2.  **DO NOT** import any libraries; `pandas` is aliased as `pd` and `plotly.express` as `px` and are already available.
3.  **DO NOT** include any comments or explanations in the generated code. Only return the executable Python code block.
4.  If the query asks for a plot, use `plotly.express` (px) and assign the plot to a variable named `fig`.
5.  If a plot is generated, ensure it has a meaningful `title` and appropriate `x` and `y` axis labels.
6.  Handle common aggregations like 'sum', 'mean', 'count', 'min', 'max' appropriately.
7.  Ensure column names in the code exactly match the column names from the data preview (case-sensitive).
8.  If the query seems impossible to answer with the given data, or requires complex operations not easily done in a single code block, return `# Cannot generate code for this query.`
9.  Prioritize creating a meaningful visualization if a visual output is implied by the query (e.g., "show trend", "distribution", "relationship").

Examples:
# User query: "Show me the total sales by region."
# Code:
# result = df.groupby('Region')['Sales'].sum().reset_index()
# fig = px.bar(result, x='Region', y='Sales', title='Total Sales by Region')

# User query: "What is the average age of customers?"
# Code:
# result = df['Age'].mean()

# User query: "Plot the relationship between Price and Quantity."
# Code:
# fig = px.scatter(df, x='Price', y='Quantity', title='Relationship between Price and Quantity')

Return only the Python code.
"""
    model = genai.GenerativeModel("gemini-pro")
    with st.spinner("🧠 QueryBuddy AI is thinking..."):
        try:
            response = model.generate_content(prompt)
            # Access response.text and clean up any markdown code blocks
            code = response.text.strip()
            if code.startswith("```python"):
                code = code[len("```python"):].strip()
            if code.endswith("```"):
                code = code[:-len("```")].strip()
            return code
        except Exception as e:
            st.error(f"Gemini API call failed: {e}")
            return "# Error generating code from Gemini."

def analyze_query(query, df):
    output = ""
    fig = None
    code = ""
    error_message = None

    # Try basic logic first (optional, but good for simple, fast answers)
    query_lower = query.lower()
    if "total" in query_lower and "per" in query_lower:
        num_col, group_col = None, None
        for col in df.columns:
            if col.lower() in query_lower and df[col].dtype in ['int64', 'float64']:
                num_col = col
            elif col.lower() in query_lower and df[col].dtype == 'object':
                group_col = col

        if num_col and group_col:
            code = f"result = df.groupby('{group_col}')['{num_col}'].sum().reset_index()\nfig = px.bar(result, x='{group_col}', y='{num_col}', title='Total {num_col} per {group_col}')"
            try:
                fig = safe_exec(code, df)
                output = f"Calculated total **{num_col}** per **{group_col}**."
            except Exception as e:
                error_message = f"Error during direct calculation: {e}"
        else: # If basic logic didn't fully match, try Gemini
            code = gpt_generate_code(query, df.head())
            if code and not code.strip().startswith("# Cannot generate code"):
                try:
                    fig = safe_exec(code, df)
                    output = "🧠 Gemini interpreted your query and generated a response."
                except Exception as e:
                    error_message = f"⚠️ Gemini generated code failed to execute.\nError: {str(e)}"
                    code = f"# Error during execution: {str(e)}\n" + code # Keep the generated code for review
            else:
                output = "🤷‍♀️ QueryBuddy AI couldn't generate a specific analysis for this question. Please try rephrasing."
                code = "# No valid code generated or could not understand query."

    else: # If no basic logic match, directly go to Gemini
        code = gpt_generate_code(query, df.head())
        if code and not code.strip().startswith("# Cannot generate code"):
            try:
                fig = safe_exec(code, df)
                output = "🧠 Gemini interpreted your query and generated a response."
            except Exception as e:
                error_message = f"⚠️ Gemini generated code failed to execute.\nError: {str(e)}"
                code = f"# Error during execution: {str(e)}\n" + code # Keep the generated code for review
        else:
            output = "🤷‍♀️ QueryBuddy AI couldn't generate a specific analysis for this question. Please try rephrasing."
            code = "# No valid code generated or could not understand query."

    return output, fig, code, error_message

def suggest_questions(df):
    questions = []
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    # Attempt to identify date columns more robustly
    date_cols = []
    for col in df.columns:
        # Check if column name contains 'date' or if pandas recognizes it as datetime after conversion attempt
        if 'date' in col.lower():
            date_cols.append(col)
        else:
            try:
                # Try converting to datetime to see if it's a date-like column
                pd.to_datetime(df[col], errors='coerce')
                if not df[col].isnull().all(): # Make sure it's not all NaT
                    date_cols.append(col)
            except:
                pass # Not a date column

    # Remove duplicates from date_cols if any
    date_cols = list(set(date_cols))

    for num in num_cols:
        # Suggest total per category
        for cat in cat_cols:
            questions.append(f"What is the total {num} per {cat}?")
        # Suggest trends over time
        for date in date_cols:
            questions.append(f"Show me the trend of {num} over {date}.")

    if len(num_cols) >= 2:
        # Suggest relationships between numerical columns
        for i in range(len(num_cols)):
            for j in range(i+1, len(num_cols)):
                questions.append(f"Is there a relationship between {num_cols[i]} and {num_cols[j]}?")

    if len(cat_cols) > 0:
        # Suggest distribution of a categorical column
        questions.append(f"What is the distribution of {cat_cols[0]}?")

    return list(dict.fromkeys(questions))[:7] # Show top N unique questions

if uploaded_file:
    try:
        # Use st.cache_data to cache DataFrame loading
        @st.cache_data
        def load_data(file):
            if file.name.endswith(".csv"):
                return pd.read_csv(file)
            else:
                return pd.read_excel(file)

        df = load_data(uploaded_file)

        st.success("✅ File uploaded successfully!")
        with st.expander("🧾 Data Preview"):
            st.dataframe(df.head())

        with st.expander("📊 Data Profiling"):
            st.write("**Shape:**", df.shape)
            st.write("**Missing Values:**")
            st.dataframe(df.isnull().sum().to_frame(name='Missing Count'))
            st.write("**Summary Stats (Numeric):**")
            st.dataframe(df.describe())
            st.write("**Summary Stats (Categorical):**")
            st.dataframe(df.select_dtypes(include='object').describe())


        st.divider()
        st.subheader("💬 Chat with your data")

        # Suggested Questions
        st.markdown("#### 💡 Suggested Questions")
        cols = st.columns(3) # Use columns for layout
        for i, q in enumerate(suggest_questions(df)):
            with cols[i % 3]: # Distribute buttons across 3 columns
                if st.button(q, key=f"suggested_q_{i}"):
                    response_text, fig, code, error = analyze_query(q, df)
                    st.session_state.chat_history.append({"role": "user", "content": q})
                    st.session_state.chat_history.append({"role": "assistant", "content": response_text, "fig": fig, "code": code, "error": error})
                    st.experimental_rerun()

        st.divider()

        # Chat History
        for i, chat in enumerate(st.session_state.chat_history):
            if chat["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(chat["content"])
            else: # role == "assistant"
                with st.chat_message("assistant"):
                    st.markdown(chat["content"])
                    if chat.get("fig"):
                        st.plotly_chart(chat["fig"])

                        # Download button for plot
                        buffer = io.BytesIO()
                        chat["fig"].write_image(buffer, format="png")
                        st.download_button(
                            label="Download Plot as PNG",
                            data=buffer,
                            file_name=f"querybuddy_plot_{i}.png",
                            mime="image/png"
                        )
                    if chat.get("code"):
                        with st.expander("🔍 View & Edit Generated Code"):
                            # Use a unique key for each code editor in history
                            editable_code = st.code_editor(
                                chat["code"],
                                language="python",
                                key=f"code_editor_{i}",
                                height=200 # Set a default height
                            )
                            if st.button(f"Run Edited Code (Message {i+1})", key=f"run_edited_code_{i}"):
                                try:
                                    with st.spinner("Executing edited code..."):
                                        new_fig = safe_exec(editable_code, df)
                                    if new_fig:
                                        st.plotly_chart(new_fig)
                                        new_buffer = io.BytesIO()
                                        new_fig.write_image(new_buffer, format="png")
                                        st.download_button(
                                            label="Download New Plot as PNG",
                                            data=new_buffer,
                                            file_name=f"querybuddy_edited_plot_{i}.png",
                                            mime="image/png"
                                        )
                                    else:
                                        st.info("Code executed, but no Plotly figure was assigned to `fig` variable.")
                                except Exception as e:
                                    st.error(f"Error executing edited code: {e}")
                    if chat.get("error"):
                        st.error(chat["error"])


        user_input = st.text_input("Type your question about the dataset:", key="user_input")

        if st.button("Ask") and user_input:
            # Add user input to chat history immediately
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            response_text, fig, code, error = analyze_query(user_input, df)
            # Add AI response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response_text, "fig": fig, "code": code, "error": error})
            st.experimental_rerun()
        elif user_input and not st.button("Ask"): # Allow pressing Enter to submit
             st.session_state.chat_history.append({"role": "user", "content": user_input})
             response_text, fig, code, error = analyze_query(user_input, df)
             st.session_state.chat_history.append({"role": "assistant", "content": response_text, "fig": fig, "code": code, "error": error})
             st.experimental_rerun()


    except Exception as e:
        st.error(f"❌ Error loading or processing file: {e}")
        st.info("Please ensure your file is a valid CSV or Excel format and try again.")
else:
    st.info("👆 Please upload a dataset to begin.")