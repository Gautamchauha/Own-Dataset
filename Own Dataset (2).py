import streamlit as st
import pandas as pd
import networkx as nx
import re
import google.generativeai as genai
from pyvis.network import Network
import tempfile
import os
import numpy as np
import random


# ✅ Configure Gemini API
from dotenv import load_dotenv

# Configure Google AI API
load_dotenv()  # Load environment variables from .env file
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# 🔹 Function to extract hierarchical dependencies from the dataset
def extract_hierarchical_dependencies(df, target_feature, max_depth=3, threshold=0.2):
    if target_feature not in df.columns:
        return {}, {}

    df_encoded = df.copy()

    # 🔹 Encode categorical features
    categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols:
        df_encoded[col] = pd.factorize(df_encoded[col])[0]

    # 🔹 Compute correlation matrix
    correlations = df_encoded.corr()
    if target_feature not in correlations:
        return {}, {}

    # 🔹 Find Primary dependencies based on correlation threshold
    corr_values = correlations[target_feature].abs()
    threshold = max(corr_values.median(), threshold)  # Dynamic threshold
    corr_values = corr_values[corr_values > threshold]

    dependencies = {target_feature: []}
    level_mapping = {target_feature: 0}

    def find_dependencies(feature, current_depth):
        if current_depth > max_depth or feature not in correlations:
            return

        sorted_features = correlations[feature].abs().sort_values(ascending=False)
        related_features = sorted_features[sorted_features > threshold].index.tolist()[1:6]

        for rel in related_features:
            if rel not in dependencies:
                dependencies[rel] = []
                level_mapping[rel] = current_depth

            dependencies[feature].append(rel)
            find_dependencies(rel, current_depth + 1)

    find_dependencies(target_feature, 1)
    return dependencies, level_mapping
# 🔹 Function to fetch AI-based dependencies
def normalize_text(text):
    return re.sub(r"\*\s{2,}", "* ", text)

# 🔹 Function to fetch AI-based dependencies dynamically based on the dataset feature context
# Fetch AI-based dependencies dynamically
def get_ai_dependencies(feature, dataset_features):
    # Build a dynamic context from the dataset features
    feature_context = ', '.join(dataset_features)

    prompt = (
        f"Given the dataset with the following features: {feature_context}, "
        "now drive the context of the dataset whats it referring to. "
        f"list at least 10 to 20 primary dependencies for the feature '{feature}'. "
        "These dependencies should be not the features in the dataset  but apart from that  which have a direct relationship or impact on the target feature. "
        "Each dependency should be formatted as:\n"
        "* **feature_name** (reason why it is a primary dependency)\n"
        "Focus only on Primary dependencies—no secondary or tertiary ones. "
        "Provide a diverse set of dependencies based on the context of these features, and ensure that they are logically related to each other."
    )

    try:
        # Query the AI model with the updated prompt
        response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
        raw_output = response.text if response.text else "EMPTY RESPONSE"

        print(f" AI Response for '{feature}':\n{raw_output}")  # Debugging Output

        if raw_output == "EMPTY RESPONSE":
            return {"Primary": [], "Explanations": {}}

        primary_dependencies = []
        explanations = {}

        # Parse the raw response into dependencies and their explanations
        for line in raw_output.split("\n"):
            line = line.strip()
            if line.startswith("*   **"):
                match = re.match(r"\*\s*\*\*([^*]+)\*\*\s*\(([^)]+)\)", line)
                if match:
                    feature_name, reason = match.groups()
                    primary_dependencies.append(feature_name.strip())
                    explanations[feature_name.strip()] = reason.strip()

        # Filter out dependencies that already exist
        existing_dependencies = set(st.session_state.dependencies.get(feature, []))
        new_primary_dependencies = [dep for dep in primary_dependencies if dep not in existing_dependencies]

        return {"Primary": new_primary_dependencies[:20], "Explanations": explanations}

    except Exception as e:
        print(f" AI Error: {e}")
        return {"Primary": [], "Explanations": {}}

# ✅ Initialize session state
if "dependencies" not in st.session_state:
    st.session_state.dependencies = {}
if "level_mapping" not in st.session_state:
    st.session_state.level_mapping = {}
if "dataset_features" not in st.session_state:
    st.session_state.dataset_features = []
if "graph_ready" not in st.session_state:
    st.session_state.graph_ready = False
if "ai_dependencies" not in st.session_state:
    st.session_state.ai_dependencies = {}
if "expanded_features" not in st.session_state:
    st.session_state.expanded_features = set()
if "df" not in st.session_state:
    st.session_state.df = None

st.title(" AI-Powered Dependency Analyzer (Dataset Mode)")

# 🔹 Step 1: Upload Dataset
uploaded_file = st.file_uploader(" Upload your dataset (CSV format)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df
    st.session_state.dataset_features = df.columns.tolist()
    st.write(" Dataset loaded successfully!")

    # 🔹 Step 2: User selects target feature
    target_feature = st.selectbox(" Select the Target Feature:", df.columns.tolist())

    if st.button("🔍 Analyze Dataset-Based Dependencies"):
        dependencies, level_mapping = extract_hierarchical_dependencies(df, target_feature)
        st.session_state.dependencies = dependencies
        st.session_state.level_mapping = level_mapping
        st.session_state.graph_ready = True
        st.session_state.expanded_features.add(target_feature)
        st.success(" Dependency graph generated!")

# 🔹 Function to render dependency graph

# 🔹 Function to set graph options
def set_graph_options(net):
    options = """
    {
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "LR",  
          "sortMethod": "directed",
          "levelSeparation": 300,
          "nodeSpacing": 200,
          "treeSpacing": 300,
          "blockShifting": false,
          "edgeMinimization": true,
          "parentCentralization": true
        }
      },
      "physics": {
        "enabled": false
      },
      "interaction": {
        "hover": true,
        "dragNodes": true,
        "dragView": true,
        "zoomView": true
      },
      "edges": {
        "color": {
          "color": "darkblue"
        },
        "width": 2.5,
        "smooth": {
          "type": "cubicBezier",
          "forceDirection": "horizontal"
        }
      },
      "nodes": {
        "font": {
          "size": 20,  
          "face": "Arial"
        },
        "shape": "box",
        "margin": 15
      }
    }
    """
    net.set_options(options)

# 🔹 Function to get color by level
def get_color_by_level(level):
    color_map = {
        0: "lightgreen",  # Target feature
        1: "lightblue",   # Level 1 dependencies
        2: "lightyellow",  # Level 2 dependencies
        3: "lightcoral",   # Level 3 dependencies
        4: "lightgray",    # Level 4 dependencies (if needed)
    }
    return color_map.get(level, "lightgray")

# 🔹 Function to render dependency graph
def render_graph():
    G = nx.DiGraph()  # Use a directed graph for unidirectional edges

    for node, level in st.session_state.level_mapping.items():
        G.add_node(node, level=level)

    for parent, children in st.session_state.dependencies.items():
        for child in children:
            G.add_edge(parent, child)  # Ensure directed edges

    net = Network(height="600px", width="100%", directed=True)  # Set directed to True

    # Set graph options
    set_graph_options(net)

    for node in G.nodes:
        level = G.nodes[node]['level']
        color = get_color_by_level(level)  # Get color based on level
        net.add_node(node, label=node, color=color, size=30, shape="box", title=node)  # Ensure nodes are boxes and set title for hover text

    for edge in G.edges:
        net.add_edge(edge[0], edge[1], color="gray", width=2)

    # ✅ Safe file handling
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_file:
        temp_filename = temp_file.name  # Store filename before closing
        net.save_graph(temp_filename)

    # ✅ Read and display the file safely
    with open(temp_filename, "r", encoding="utf-8") as f:
        html_content = f.read()
        st.components.v1.html(html_content, height=600, scrolling=True)

    with open(temp_filename, "r") as f:
        st.download_button("Download Graph as HTML", data=f, file_name="dependency_graph.html", mime="text/html")

    # ✅ Delete file safely after rendering
    try:
        os.remove(temp_filename)
    except PermissionError:
        print(f" Warning: Could not delete temp file {temp_filename}. It may still be in use.")

# Call render_graph() when the graph is ready
if st.session_state.graph_ready:
    st.write("## Dependency Graph")
    render_graph()
    # Debugging: Check if dependencies are populated
    st.write("Available Features for Expansion:", list(st.session_state.dependencies.keys()))

    # After selecting the feature to expand
# After selecting the feature to expand
selected_feature = st.selectbox(" Select a feature to expand:", list(st.session_state.dependencies.keys()))
st.write("Selected Feature:", selected_feature)

# Proceed with AI suggestion if a feature is selected
if selected_feature:
    # Check if dependencies have already been loaded for the selected feature
    if selected_feature not in st.session_state.ai_dependencies:
        # Pass both selected_feature and dataset_features to the AI function
        ai_data = get_ai_dependencies(selected_feature, st.session_state.dataset_features)  # Fixed
        st.session_state.ai_dependencies[selected_feature] = ai_data  # Store AI data

    # Retrieve the AI-generated dependencies for the selected feature
    ai_dependency_data = st.session_state.ai_dependencies.get(selected_feature, {"Primary": []})
    suggested_deps = ai_dependency_data["Primary"]

    # Display the primary dependencies with their explanations
    if suggested_deps:
        st.subheader(f"Primary Dependencies for {selected_feature} (with Explanations):")
        for dep in suggested_deps:
            explanation = ai_dependency_data["Explanations"].get(dep, "No explanation available.")
            st.markdown(f"**{dep}:** {explanation}")

        # Allow users to select dependencies
        selected_suggestions = st.multiselect(" Select AI-suggested dependencies:", suggested_deps)

        if st.button(f" Confirm Dependencies for {selected_feature}"):
            if selected_suggestions:
                # Adding selected suggestions to the existing dependencies
                st.session_state.dependencies[selected_feature].extend(selected_suggestions)
                # Add selected suggestions to the expanded features so they can be used for future expansion
                st.session_state.expanded_features.update(selected_suggestions)
                # Also update the available features for expansion
                st.session_state.graph_ready = True
                st.success(f" Dependencies for '{selected_feature}' added!")

                # Add the newly confirmed dependencies to the list for recursive expansion
                st.session_state.dependencies[selected_feature].extend(selected_suggestions)

                # **Update the select dropdown list to include newly added features**
                st.session_state.dataset_features.extend(selected_suggestions)  # Add to the list of available features

                # Re-run the Streamlit app to update the UI with new dependencies
                st.rerun()
    else:
        st.write("No AI-suggested dependencies available.")

def generate_expanded_dataset():
    # Get the current dataset and the expanded features
    df = st.session_state.df
    expanded_features = list(st.session_state.expanded_features)

    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        st.stop()

    # Get the current columns in the dataset
    existing_columns = set(df.columns.tolist())
    
    # Filter out features that already exist in the dataset
    new_features = [feature for feature in expanded_features if feature not in existing_columns]
    
    # Add only the new features to the dataset
    for feature in new_features:
        # Generate the data for the new feature based on its dependency level
        # Placeholder function for generating feature values. You can modify this based on your logic.
        df[feature] = np.random.uniform(0, 1, size=len(df))  # Replace with actual value generation logic

    # Display the updated dataframe with the expanded features
    st.write("Updated Dataset with Expanded Features:")
    st.dataframe(df)
    

    # Option to download the expanded dataset
    if st.button("Download Expanded Dataset"):
        # Save the dataframe to a CSV file and provide download link
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            expanded_csv_path = tmp_file.name
            df.to_csv(expanded_csv_path, index=False)
            st.download_button("Download CSV", data=open(expanded_csv_path, "rb"), file_name="expanded_dataset.csv")
            # Footer with Copyright Information


# Call the function to generate the expanded dataset and allow downloading
generate_expanded_dataset()
st.markdown(
    """
    <hr style="border:1px solid gray;margin-top:20px;margin-bottom:10px;">
    <div style="text-align:center;">
        <p>© 2025 All rights reserved.</p>
        <p>Developed by Himistu Lab.</p>
    </div>
    """,
    unsafe_allow_html=True
)

