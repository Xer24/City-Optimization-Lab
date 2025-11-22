# City Optimization Lab

A modular urban simulation and optimization environment built in Python for exploring how population, zoning, transit infrastructure, and energy constraints shape the behavior of a city. The project integrates simulation, optimization, machine learning, and data visualization as a unified analytical pipeline.

## 🌆 Project Summary

City Optimization Lab is a configurable environment where users can:

- Adjust **population growth**, **zoning**, and **transit routes**
- Simulate **traffic flow**, **energy demand**, and **grid load**
- Run **optimization routines** (min traffic congestion, min energy cost)
- Cluster the city into functional **zones** using ML
- Visualize results through **heatmaps**, **energy curves**, and **city maps**

This project demonstrates applied data science, operations research, and system modeling in an urban-analytics context.

## 🧰 Tech Stack

**Python (primary)**
- NumPy, Pandas, SciPy
- scikit-learn
- PuLP / Pyomo (optimization)
- Matplotlib & Plotly
- Streamlit (UI)
- SQLAlchemy (database layer)

**SQL**
- SQLite (local development)
- PostgreSQL (optional "production" mode)

**MATLAB**
- Optimization Toolbox for benchmarking
- Additional prototypes stored in `matlab_benchmarks/`

## 🚀 Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the core simulation

```bash
python src/main.py
```

### 3. Launch the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

## 📊 Features

### ✔ Initial Milestone
- Basic city grid representation, with RNG roads
- Prototyping traffic + energy models
- Initial ML clustering pipeline
- Database schema for scenarios and runs

### 🔧 In Progress
- Full 24-hour simulation loop
- Optimization engine (LP and MIP versions)
- Visualization dashboard

### 🌐 Future Extensions
- Multi-objective optimization (Pareto frontier)
- Transit routing solver with OR-Tools
- Stochastic demand models

## 📫 Contact

Project maintained by **Mikail Durrani**.
