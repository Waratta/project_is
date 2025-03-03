import streamlit as st

Machine_Learning_page = st.Page("Machine_Learning.py", title="Machine Learning", icon=":material/circle:")
Neural_Network_page = st.Page("Neural_Network.py", title="Neural Network", icon=":material/circle:")
demo_Machine_Learning_page = st.Page("demo_ML.py", title="demo Machine Learning", icon=":material/circle:")
demo_Neural_Network_page = st.Page("demo_N.py", title="demo Neural Network", icon=":material/circle:")

pg = st.navigation([Machine_Learning_page, Neural_Network_page,demo_Machine_Learning_page,demo_Neural_Network_page])
pg.run()
