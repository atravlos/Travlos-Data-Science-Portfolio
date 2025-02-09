import streamlit as st
import pandas as pd

# Introduction
st.title("Find Your Penguin!")
st.write("This app allows you to search for penguins based on species and bill length.")

# Load in and display dataframe
df = pd.read_csv("data/penguins.csv")
st.write("Full List of Penguin Characters:")
st.dataframe(df)

# Create interactive filtering selectbox based on species
species = st.selectbox("Select a species", df["species"].unique())

species_df = df[df["species"] == species]

st.write(f"Penguins that are {species} species:")
st.dataframe(species_df)

# Create interactive filtering slider based on bill length.

bill_length = st.slider("Choose a bill length range:", 
                   min_value = df["bill_length_mm"].min(),
                   max_value = df['bill_length_mm'].max())


st.write(f"Penguins with bill length under {bill_length}:")
st.dataframe(species_df[species_df['bill_length_mm'] <= bill_length])