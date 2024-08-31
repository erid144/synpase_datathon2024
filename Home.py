import sys
sys.path.append('src')
sys.path.append('models')
sys.path.append('data')
from librerias import *
from utils import *


st.set_page_config(page_title="Analisis de Datos")

# Ahora puedes agregar el resto de tu código de Streamlit
st.title("Bienvenido a mi aplicación de Streamlit")


# Mostrar un mensaje de bienvenida
st.title("Analisis del Régimen Contributivo en el Seguro Familiar de Salud Dominicano")

# Mostrar un subtítulo
st.header("Jesus Diaz")

# Mostrar un párrafo informativo
st.write("Este proyecto tiene como objetivo realizar un análisis exhaustivo de las Administradoras de Riesgos de Salud (ARS) en el marco del Seguro Familiar de Salud (SFS) en la República Dominicana. Se pretende permitir a los usuarios filtrar, ordenar, agrupar y explorar los datos de forma intuitiva para obtener una comprensión más profunda de este importante sector de la salud en el país.")



st.session_state.dataframes,st.session_state.nombres_especificos,st.session_state.column_descriptions = load_dataframes()
