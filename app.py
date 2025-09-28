import streamlit as st
import pandas as pd
import numpy as np

st.title("Storm Restoration Optimization Assistant - Test Build")
st.write("✅ App successfully deployed!")

data = pd.DataFrame(np.random.randn(10, 3), columns=["A", "B", "C"])
st.line_chart(data)
