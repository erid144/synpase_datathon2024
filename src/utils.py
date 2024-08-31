from librerias import *
import os
def extract_numeric_part(flight_number):
    """
    Extracts the numeric part from a flight number string.
    
    Args:
        flight_number (str): A flight number string containing alphanumeric characters.
    
    Returns:
        int: The numeric part of the flight number, or None if the flight_number is not a string or has no numeric part.
    """   
    if isinstance(flight_number, str):
        numeric_part = re.findall(r'\d+', flight_number)
        return int(numeric_part[0]) if numeric_part else None
    else:
        return None

def extract_alphabetic_part(flight_number):
    """
    Extracts the alphabetic part from a flight number string.
    
    Args:
        flight_number (str): A flight number string containing alphanumeric characters.
    
    Returns:
        str: The alphabetic part of the flight number, or None if the flight_number is not a string or has no alphabetic part.
    """
    if isinstance(flight_number, str):
        alphabetic_part = re.findall(r'[a-zA-Z]+', flight_number)
        return alphabetic_part[0] if alphabetic_part else None
    else:
        return 0


def plot_flight_analysis(df):
    """
    Plots various graphs for flight data analysis, including:
    1. Share of International and National Departing Flights by Airline
    2. Top 25 Destinations from SCL
    3. Heatmap of Flight Count by Month and Day of the Week
    4. Distribution of flights by the hour of the day
    5. Distribution of international and national flights by month
    
    Args:
    df (pd.DataFrame): The flight data DataFrame with the following columns:
                       DIA, SIGLADES, OPERA, TIPOVUELO, DIANOM, MES
    
    Returns:
    None
    """
    title_font = dict(family="Arial, sans-serif", size=18, color="black")
    axis_font = dict(family="Arial, sans-serif", size=14, color="black")    
    airline_type_pivot = pd.pivot_table(data=df, values='DIA', index=['OPERA'], columns=['TIPOVUELO'], aggfunc='count', fill_value=0)
    airline_type_percentage = airline_type_pivot / airline_type_pivot.sum()
    airline_type_percentage = airline_type_percentage.reset_index()
    flight_types = [('I', 'International'), ('N', 'National')]
    #1. Share of International and National Departing Flights by Airline
    for flight_type, flight_label in flight_types:
        sorted_data = airline_type_percentage.sort_values(flight_type, ascending=False)
        fig1 = px.bar(sorted_data, x='OPERA', y=flight_type, title=f"Share of {flight_label} Departing Flights by Airline",
                     labels={'OPERA': 'Airline', flight_type: f'Share of {flight_label} Flights'},
                     color=flight_type, color_continuous_scale='Viridis')
        fig1.update_yaxes(tickformat=".0%", title='Share of Flights')
        fig1.update_xaxes(title='Airline')
        fig1.update_layout(
            title_font=title_font,
            xaxis_title_font=axis_font,
            yaxis_title_font=axis_font,
            coloraxis_colorscale="Viridis"
        )
        pyo.iplot(fig1)
    #2. Top 25 Destinations from SCL
    top_destinations = df['SIGLADES'].value_counts().iloc[:25]
    fig2 = px.bar(top_destinations, x=top_destinations.index, y=top_destinations.values,
                 title="Top 25 Destinations from SCL",
                 labels={'x': 'Destination City', 'y': 'Number of Flights'},
                 color=top_destinations.values, color_continuous_scale='Viridis')
    fig2.update_xaxes(title='Destination City')
    fig2.update_xaxes(tickangle=90)
    fig2.update_yaxes(title='Number of Flights')
    fig2.update_layout(
    title_font=title_font,
    xaxis_title_font=axis_font,
    yaxis_title_font=axis_font,
    coloraxis_colorscale="Viridis"
    )
    pyo.iplot(fig2)
    
    #3. Heatmap of Flight Count by Month and Day of the Week
    heatmap_data = df.pivot_table(values='DIA', index='DIANOM', columns='MES', aggfunc='count', fill_value=0)
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    heatmap_data.columns = [month_names[col - 1] for col in heatmap_data.columns]
    fig3 = px.imshow(heatmap_data, title='Heatmap of Flight Count by Month and Day of Week',
                     labels=dict(x='Month', y='Day of Week', color='Number of Flights'))
    fig3.update_xaxes(side='bottom', tickangle=-45)
    fig3.update_layout(
        title_font=title_font,
        xaxis_title_font=axis_font,
        yaxis_title_font=axis_font,
        coloraxis_colorscale="Viridis"
    )
    pyo.iplot(fig3)
    
    #4. Distribution of flights by the hour of the day
    df['Hour'] = pd.to_datetime(df['Fecha-O']).dt.hour
    flights_by_hour = df['Hour'].value_counts().sort_index()
    fig4 = px.bar(flights_by_hour, x=flights_by_hour.index, y=flights_by_hour.values,
                 title="Flights Distribution by Hour of the Day",
                 labels={'x': 'Hour of the Day', 'y': 'Number of Flights'},
                 color=flights_by_hour.values, color_continuous_scale='Viridis')
    fig4.update_xaxes(title='Hour of the Day')
    fig4.update_yaxes(title='Number of Flights')
    fig4.update_xaxes(tickangle=0)
    fig4.update_layout(
    title_font=title_font,
    xaxis_title_font=axis_font,
    yaxis_title_font=axis_font,
    coloraxis_colorscale="Viridis"
    )
    pyo.iplot(fig4)

    # 5. Distribution of international and national flights by month
    flights_by_month_type = pd.pivot_table(df, values='DIA', index=['MES'], columns=['TIPOVUELO'], aggfunc='count', fill_value=0)
    flights_by_month_type = flights_by_month_type.reset_index()

    # Convert month numbers to month names
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    flights_by_month_type['MES'] = flights_by_month_type['MES'].apply(lambda x: month_names[x - 1])

    fig5 = px.bar(flights_by_month_type, x='MES', y=['I', 'N'],
                  title="Distribution of International and National Flights by Month",
                  labels={'MES': 'Month', 'value': 'Number of Flights', 'variable': 'Flight Type'},
                  color_discrete_sequence=px.colors.qualitative.Plotly)
    fig5.update_xaxes(title='Month')
    fig5.update_xaxes(tickangle=0)
    fig5.update_yaxes(title='Number of Flights')
    fig5.update_layout(
        title_font=title_font,
        xaxis_title_font=axis_font,
        yaxis_title_font=axis_font
    )
    pyo.iplot(fig5)



def plot_delay_rate_by_group(df, grouping_column):
    """
    Generates a bar graph with the delay rate of flights in *df* by *grouping_column*.
    
    :param df: Data to be plotted. Must have "atraso_15" and *grouping_column* among columns.
    :type df: DataFrame
    :param grouping_column: Column of *df* to group delay rates by.
    :type grouping_column: str
    """
    # Calculate the mean delay rate for each group
    group_means = df.groupby(grouping_column, as_index=False).mean()[[grouping_column, 'atraso_15']]
    group_means['atraso_15'] *= 100

    # Calculate the global mean delay rate
    global_mean = np.nanmean(df['atraso_15']) * 100
    if grouping_column == 'MES':
        month_names = {
            1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
            7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
        }
        group_means[grouping_column] = group_means[grouping_column].map(month_names)


    # Sort the data by delay rate
    sorted_means = group_means.sort_values('atraso_15', ascending=False)

    # Create the barplot with seaborn
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=sorted_means, x=grouping_column, y='atraso_15', ax=ax, order=sorted_means[grouping_column])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.axhline(global_mean, color='r', linestyle='--', label='Media Global')
        ax.set_title(f"Proporción de Vuelos con Retraso por {grouping_column}")
        ax.set_ylabel("Proporción de Vuelos con Retraso")
        ax.set_xlabel(grouping_column)
        ax.legend()
        plt.show()
        
        
def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=False, title='Confusion Matrix', cmap='Blues'):
    """
    Plots a confusion matrix using seaborn heatmap.
    
    :param y_true: True labels.
    :type y_true: array-like, shape (n_samples,)
    :param y_pred: Predicted labels.
    :type y_pred: array-like, shape (n_samples,)
    :param labels: List of labels to index the matrix.
    :type labels: list of str, optional, default: None
    :param normalize: Whether to normalize the confusion matrix.
    :type normalize: bool, optional, default: False
    :param title: Title for the confusion matrix plot.
    :type title: str, optional, default: 'Confusion Matrix'
    :param cmap: Colormap to be used for the heatmap.
    :type cmap: str, optional, default: 'Blues'
    """
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Normalize the confusion matrix if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create a seaborn heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap, xticklabels=labels, yticklabels=labels)

    # Set plot title and labels
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Show the plot
    plt.show()
    
    
def find_best_threshold(y_test, y_pred_prob):
    """
    Finds the best threshold for classification based on the highest F1 score.
    
    :param y_test: True target values.
    :type y_test: array-like
    :param y_pred_prob: Predicted probabilities for the positive class.
    :type y_pred_prob: array-like
    :return: Best threshold.
    :rtype: float
    """
    thresholds = np.arange(0, 1.01, 0.01)
    best_threshold = 0
    best_f1_score = 0
    for t in thresholds:
        y_pred_t = (y_pred_prob >= t).astype(int)
        f1 = f1_score(y_test, y_pred_t)
        if f1 > best_f1_score:
            best_f1_score = f1
            best_threshold = t
    return best_threshold

def plot_roc_curve(fpr, tpr, roc_auc, classifier_name):
    """
    Plots the ROC curve for a given classifier.
    
    :param fpr: False positive rates.
    :type fpr: array-like
    :param tpr: True positive rates.
    :type tpr: array-like
    :param roc_auc: Area under the ROC curve.
    :type roc_auc: float
    :param classifier_name: Name of the classifier.
    :type classifier_name: str
    """
    plt.plot(fpr, tpr, label=f'{classifier_name} AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

def train_with_cross_validation(clf, X, y, cv=5):
    """
    Trains a classifier using cross-validation and returns the average performance metrics and the trained classifier.

    :param clf: Classifier to be trained.
    :type clf: Classifier object
    :param X: Training data.
    :type X: DataFrame
    :param y: Labels for the training data.
    :type y: Series
    :param cv: Number of cross-validation folds.
    :type cv: int
    :return: A dictionary containing average performance metrics and the trained classifier.
    :rtype: dict
    """

    y_pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)

    avg_metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_pred)
    }

    clf.fit(X, y)  # Train the classifier on the entire dataset

    return avg_metrics, clf

def optimize_hyperparameters(classifier, param_grid, X_train, y_train,cv):
    """
    Optimizes hyperparameters for a classifier using GridSearchCV.
    
    :param classifier: The classifier to optimize.
    :type classifier: estimator
    :param param_grid: Dictionary with parameters names as keys and lists of parameter settings to try as values.
    :type param_grid: dict
    :param X_train: Training data.
    :type X_train: array-like
    :param y_train: Training target values.
    :type y_train: array-like
    :return: Best estimator found by grid search.
    :rtype: estimator
    """
    grid_search = GridSearchCV(classifier, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1,verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def encode_categorical_columns(data, categorical_columns):
    """
    Encodes categorical columns in *data* using One-Hot Encoding.
    
    :param data: Data to be encoded.
    :type data: DataFrame
    :param categorical_columns: Categorical columns in *data*.
    :type categorical_columns: list of str
    :return: Encoded data.
    :rtype: DataFrame
    """
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    categorical_data = data[categorical_columns]
    encoded_categorical_data = encoder.fit_transform(categorical_data)
    
    encoded_columns = []
    for cat_col, categories in zip(categorical_columns, encoder.categories_):
        encoded_columns.extend([f"{cat_col}_{category}" for category in categories])
    
    encoded_df = pd.DataFrame(encoded_categorical_data, columns=encoded_columns, index=data.index)
    data = data.drop(categorical_columns, axis=1)
    data = pd.concat([data, encoded_df], axis=1)
    
    return data

def train_and_evaluate(clf, X_train, y_train, X_test, y_test):
    """
    Trains and evaluates a classifier using the specified data.
    
    :param clf: Classifier to be trained.
    :type clf: Classifier object
    :param X_train: Training data.
    :type X_train: DataFrame
    :param y_train: Labels for the training data.
    :type y_train: Series
    :param X_test: Test data.
    :type X_test: DataFrame
    :param y_test: Labels for the test data.
    :type y_test: Series
    """
    # Train the classifier
    clf.fit(X_train, y_train)
    
    # Predict on the test data
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    
    # Print evaluation metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f"F1-score: {f1_score(y_test, y_pred)}")
    print(f"AUC: {roc_auc_score(y_test, y_pred_prob)}")
    print(classification_report(y_test, y_pred))
    
    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_pred_prob):.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='k')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def generate_requirements_txt(filename="requirements.txt"):
    """
    Generates a `requirements.txt` file containing a list of all installed packages in the current environment.

    Parameters:
    -----------
    filename : str, optional (default="requirements.txt")
        The name of the file to be generated, without any path information.

    Returns:
    --------
    None
    """

    # run pip freeze command to get a list of installed packages
    installed_packages = subprocess.check_output(['pip', 'freeze']).decode('utf-8').split('\n')

    # create the data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # open the specified requirements.txt file in write mode within the data directory
    with open(f"data/{filename}", 'w') as f:
        # write each package to the file
        for package in installed_packages:
            f.write(package + '\n')

    print(f"Generated {filename} successfully in the data directory!")
    
def install_requirements(filepath='requirements.txt'):
    """
    Reads the packages listed in a `requirements.txt` file and installs them in the current environment.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """

    try:
        # read the packages listed in the requirements.txt file
        with open(f"data/{filepath}", 'r') as f:
            packages = f.read().splitlines()

        # install the packages using pip
        result = subprocess.run([sys.executable, "-m", "pip", "install", *packages], capture_output=True)

        if result.returncode == 0:
            print("All packages already installed.")
        else:
            print("Installed packages successfully!")
    except FileNotFoundError:
        print("No requirements.txt file found. Continuing with the execution...")


    
def Normalizar_fecha(df,CPeriodo='Periodo de Cobertura',CAno='Ano de Cobertura'):
    """
    Toma la columna de fecha del dataframey evalua
    Si existe una columna periodo de cobertura:
        crea dos columnas auxiliares año y mes
        y con ellas formatea la columna de fecha en tipo de dato fecha.
    Si existe la columna Ano de Cobertura:
       Extrae el año eliminando otros caracteres
    Parameters:
    -----------
    df: pandas dataframe
    Cperiodo: Nombre de la columna Periodo 
    Cano: Nombre de la columna del año
    
    Returns:
    --------
    None
    """
    if CPeriodo in df.columns:
        df[CPeriodo] = df[CPeriodo].astype(str) 
        # Extraer numericos
        mask = df[CPeriodo].str.contains(r'\D', na=False)
        df.loc[mask,CPeriodo] = df.loc[mask, CPeriodo].str.replace(r'\D', '', regex=True)
        # Convertir la columna a tipo numérico si es necesario
        df[CPeriodo] = pd.to_numeric(df[CPeriodo], errors='coerce')
        df[CPeriodo] = df[CPeriodo].astype(int)
        # Separar year y month
        df['Year'] = df[CPeriodo] // 100
        df['Month'] = df[CPeriodo] % 100

        # Convertir a formato de fecha
        df[CPeriodo] = pd.to_datetime(df['Year'].astype(str) + df['Month'].astype(str), format='%Y%m')
        
    if CAno in df.columns:
        df[CAno] = df[CAno].astype(str) 
        # Extraer solo el año de cobertura en formato YYYY
        mask = df[CAno].str.contains(r'\D', na=False)
        df.loc[mask, CAno] = df.loc[mask, CAno].str.replace(r'\D', '', regex=True)
        df[CAno] = df[CAno].str[:4]
        # Convertir la columna a tipo numérico si es necesario
        df[CAno] = pd.to_numeric(df[CAno], errors='coerce')
        
def Extraer_Afiliados_Edad_Sexo(data_path, header_=5, rows_=400-205, archivocsv='data.csv'):
    
    df_header = pd.read_excel(data_path, header=5, nrows=1)# para el encabezado
    df_header.columns.values[:2] = ['Periodo de Cobertura', 'Total Afiliados']

    # Seleccionar datos de la fila 205 a la 400 Hombres
    df_data1 = pd.read_excel(data_path, header=None, skiprows=204, nrows=400-205, usecols=list(range(20)))

    # Seleccionar datos de la fila 402 a la 597 Mujeres
    df_data2 = pd.read_excel(data_path, header=None, skiprows=401, nrows=597-402, usecols=list(range(20)))


    # Concatenar los datos
    df = pd.concat([df_data1, df_data2])
    df.reset_index(drop=True, inplace=True)
    
    # Especificar los nombres de las primeras dos columnas
    df.columns = df_header.columns[:20]

    Normalizar_fecha(df)
    # Crear la columna 'Sexo' y asignar valores 'M' y 'F'
    df = df.assign(Sexo=['M'] * 195 + ['F'] * 195)
    df.to_csv(archivocsv, index=False)
    #return df

def Extraer_TasaDependencia_RC_ARS(data_path, header_=11, rows_=208-12, archivocsv='data.csv'):

    df_header = pd.read_excel(data_path, header=11, nrows=208-12,usecols="A:I")
    # Lista de nombres de columna
    column_names = [
    'Periodo de Cobertura',
    'Total_RC_Tasa_Dependencia',
    'Total_RC_Tasa_Dep_Directa',
    'ARS_AutoGest_Tasa_Dep_Directa',
    'ARS_AutoGest_Tasa_Dependencia',
    'ARS_Priv_Tasa_Dep_Directa',
    'ARS_Priv_Tasa_Dependencia',
    'ARS_Pub_Tasa_Dep_Directa',
    'ARS_Pub_Tasa_Dependencia',
    ]

    df =df_header
    df.reset_index(drop=True, inplace=True)
    
    df.columns = column_names
    Normalizar_fecha(df)
    
    df.to_csv(archivocsv, index=False)
    #return df

def Extraer_AfiliadosCotizantes_Edad_Sexo(data_path, header_=6, rows_=401-206, archivocsv='data.csv'):
    
    df_header = pd.read_excel(data_path, header=6, nrows=2, usecols="A:R")# para el encabezado
    df_header.columns.values[:2] = ['Periodo de Cobertura', 'Total Cotizantes']

    # Seleccionar datos  Hombres
    df_data1 = pd.read_excel(data_path, header=None, skiprows=205, nrows=401-206, usecols="A:R")

    # Seleccionar datos  Mujeres
    df_data2 = pd.read_excel(data_path, header=None, skiprows=402, nrows=598-403, usecols="A:R")


    # Concatenar los datos
    df = pd.concat([df_data1, df_data2])
    df.reset_index(drop=True, inplace=True)
    # Especificar los nombres de las primeras dos columnas
    df.columns = df_header.columns[:20]

    Normalizar_fecha(df)
    # Crear la columna 'Sexo' y asignar valores 'M' y 'F'
    df = df.assign(Sexo=['M'] * 195 + ['F'] * 195)
    
    df.to_csv(archivocsv, index=False)
    #return df

def Extraer_AfiliadosNoCotizantes_Edad_Sexo(data_path, header_=4, rows_=400-205, archivocsv='data.csv'):
    
    df_header = pd.read_excel(data_path, header=4, nrows=1)# para el encabezado
    df_header.columns.values[:2] = ['Periodo de Cobertura', 'Total Afiliados']

    # Seleccionar datos de la fila 205 a la 400 Hombres
    df_data1 = pd.read_excel(data_path, header=None, skiprows=204, nrows=400-205, usecols=list(range(20)))

    # Seleccionar datos de la fila 402 a la 597 Mujeres
    df_data2 = pd.read_excel(data_path, header=None, skiprows=401, nrows=597-402, usecols=list(range(20)))


    # Concatenar los datos
    df = pd.concat([df_data1, df_data2])
    df.reset_index(drop=True, inplace=True)
    # Especificar los nombres de las primeras dos columnas
    df.columns = df_header.columns[:20]

    Normalizar_fecha(df)
    # Crear la columna 'Sexo' y asignar valores 'M' y 'F'
    df = df.assign(Sexo=['M'] * 195 + ['F'] * 195)
    df.to_csv(archivocsv, index=False)
    #return df

def Extraer_Tabla_Regiones(data_path, header_=6, rows_=55-7, archivocsv='data.csv'):
    

    df = pd.read_excel(data_path, header=6, nrows=55-7, usecols="A:C")# para el encabezado


    df = df.iloc[2:-1]

    # Rellenar los valores vacíos en la columna 'Provincia' con los valores correspondientes en la columna 'Región Salud'
    df['Provincia'] = df['Provincia'].fillna(df['Región Salud'])
    df = df.ffill()
    df.reset_index(drop=True, inplace=True)
    df.to_csv(archivocsv, index=False)
    #return df

    
def Extraer_datos_regionales(datapath, header_=7, rows_=54-7, archivocsv='data.csv'):
    df = pd.read_excel(datapath, header=header_, nrows=rows_, usecols="C:GQ")
    df.columns.values[0] ="Provincia"

    # Valores faltantes para rellenar en orden
    valores_faltantes = [
        'Total general',  #0
        'Total Región Distrito Nacional',#1
        'Total Región Este',#6
        'Total Región Norte',#13
        'Total Región Sur',# 32
        'No Especificada'# 46
    ]


    # Rellenar los valores faltantes en la columna 'Provincia'
    df.loc[df['Provincia'].isna(), 'Provincia'] = valores_faltantes
    df=df.T
    df = df.set_axis(df.iloc[0], axis=1)
    df = df.iloc[1:]
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Mes'}, inplace=True)
    # Asignar None al nombre del índice
    df.columns.name = None
    df.reset_index(drop=True, inplace=True)
    # Convertir la columna 'Mes' a tipo de datos de cadena (string)

    df['Month'] = df['Mes'].apply(lambda x: re.match(r'(\w+)(?:\s*\.\s*)?(\d+)?', str(x)).group(1).lower())
    


     # Mapear el nombre del mes a su número correspondiente
    month_mapping = {
        'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12,
        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
        'julio': 7, 'agosto': 8
    }

    # Aplicar el mapeo para obtener el número del mes
    df['Month'] = df['Month'].map(month_mapping)
    df['Year'] = 2007
    # Incrementar el valor de 'Year' cada 12 registros a partir del quinto registro
    df.loc[4:, 'Year'] = (df.index[4:] -4) // 12 + 2008

    del df['Mes']
    # Crear la columna 'Periodo de Cobertura' como tipo fecha
    df['Periodo de Cobertura'] = pd.to_datetime(df['Year'].astype(str) + df['Month'].astype(str), format='%Y%m')

    # Guardar los datos en un archivo CSV
    df.to_csv(archivocsv, index=False)
    #return df
    
def Extraer_SFS_Afiliacion_Tasa_Regimen(datapath, header_=11, rows_=208-12, archivocsv='data.csv'):
   
    df_header = pd.read_excel(datapath, header=11, nrows=208-12,usecols="A:E")
    # Lista de nombres de columna
    column_names = [
    'Periodo de Cobertura',
    'SFS',
    'Tasa_Dep_Regimen_Subsidiado',
    'RC_Tasa_Dependencia',
    'RC_Tasa_Dependencia_Directa',
    ]
    df =df_header# pd.concat([df_data1, df_data2])
    df.reset_index(drop=True, inplace=True)
    # Especificar los nombres de las primeras dos columnas
    df.columns = column_names
    Normalizar_fecha(df)
    df.to_csv(archivocsv, index=False)
    #return df
    
def Extraer_SFS_Regimen_Sexo(datapath, header_=7, rows_=212-8, archivocsv='data.csv'):
   
    df_header = pd.read_excel(datapath, header=7, nrows=212-8,usecols="A:J")
    # Lista de nombres de columna
    column_names = [
    'Periodo de Cobertura',
    'SFS_Total',
    'SFS_Hombres',
    'SFS_Mujeres',
    'Reg_Subsidiado_Total',
    'Reg_Subsidiado_Hombres',
    'Reg_Subsidiado_Mujeres',
    'RC_Total',
    'RC_Hombres',
    'RC_Mujeres',  
    ]
    df =df_header
    df.reset_index(drop=True, inplace=True)
    
    df.columns = column_names
    Normalizar_fecha(df)
    df.to_csv(archivocsv, index=False)
    #return df


def Extraer_SFS_Porcentaje_Regimen(datapath, header_=6, rows_=211-7, archivocsv='data.csv'):
    
    df_header = pd.read_excel(datapath, header=6, nrows=211-7,usecols="A:G")
    # Lista de nombres de columna
    column_names = [
    'Periodo de Cobertura',
    'Poblacion Total proyectada',
    'Porcentaje de Población Cubierta por el SFS',
    'Total Seguro Familiar de Salud',
    'Reg_Subsidiado',
    'Reg_Contributivo',
    'Reg_PensionadosJub',
 
    ]
    df =df_header# pd.concat([df_data1, df_data2])
    df.reset_index(drop=True, inplace=True)
    # Especificar los nombres de las primeras dos columnas
    df.columns = column_names
    Normalizar_fecha(df)
    df.to_csv(archivocsv, index=False)
    #return df

def Extraer_FinanciamientoDispersado_TipoAfiliado(datapath, header_=7, rows_=204-7, skip=0, columns_="A:M", column_names=[], archivocsv='data.csv'):
    if not column_names:
        column_names = [
            'Periodo de Cobertura',
            'Total_Capitas_Dispersadas',
            'Total_Capitas_Dispersada_mes',
            'Total_Capitas_Dispersada_posterior',
            'Titulares_Total',
            'Titulares_Dispersadas_mes',
            'Titulares_Dispersadas_posterior',
            'Dependientes_Total',
            'Dependientes_Dispersadas_mes',
            'Dependientess_Dispersadas_posterior',
            'Adicionales_Total',
            'Adicionales_Dispersadas_mes',
            'Adicionales_Dispersadas_posterior'
        ]

    df_header = pd.read_excel(datapath, header=header_, nrows=2, usecols=columns_)
    df = pd.read_excel(datapath, header=None, skiprows=skip, nrows=rows_, usecols=columns_)
    df.columns = column_names
  
    Normalizar_fecha(df)
    df.to_csv(archivocsv, index=False)
    # return df
    
def Extraer_FinanciamientoAnual(datapath, header_=7, rows_=204-7, skip=0, columns_="A:M", column_names=[], archivocsv='data.csv'):
    if not column_names:
        column_names = [
            'Ano cobertura',
            'Total_Salud (A)',
            'PDSS_Subsidiado (A)',
            'PDSS_Contributivo (A)',
            'Otros_Planes_de_Salud (A)',
            'PIB/Precios Corrientes (B)',
            'Total en Relacion al PIB (A/B)',
            'PDSS_Subsidiado en Relación al PIB (A/B)',
            'PDSS_Contributivo en Relacion al PIB (A/B)',
            'Otros_Planes_de_Salud en Relacion al PIB (A/B)'
        ]

    df_header = pd.read_excel(datapath, header=header_, nrows=2, usecols=columns_)
    df = pd.read_excel(datapath, header=None, skiprows=skip, nrows=rows_, usecols=columns_)
    
    df.columns = column_names
    
    
    Normalizar_fecha(df)
    df.to_csv(archivocsv, index=False)
    #return df
    
    
def Extraer_PrestacionesPBS(datapath, header_=7, rows_=230-7, skip=7, columns_="A:E", column_names=[], archivocsv='data.csv'):
    if not column_names:
        column_names = [
            'Ano de Cobertura',
            'Grupo Numero',
            'Grupo Descripcion ',
            'Montos Pagados',
            'Distribucion Porcentual ',
        ]

    df_header = pd.read_excel(datapath, header=header_, nrows=2, usecols=columns_)
    df = pd.read_excel(datapath, header=None, skiprows=skip, nrows=rows_, usecols=columns_)
    
    df.columns = column_names
     # Rellenar los valores faltantes en 'Grupo Número' con 0
    df['Grupo Numero'] = df['Grupo Numero'].fillna(0)
    
    # Rellenar los valores faltantes en 'Grupo Descripción' con el texto en la columna 'Año de Cobertura'
    df['Grupo Descripcion '] = df['Grupo Descripcion '].fillna(df['Ano de Cobertura'])
    
    # Extraer solo el año de cobertura en formato YYYY
    Normalizar_fecha(df)
    
    df.to_csv(archivocsv, index=False)
    #return df
    
def Extraer_SiniestralidadPBS(datapath, header_=9, rows_=204-9, skip=9, columns_="A:D", column_names=[], archivocsv='data.csv'):
    if not column_names:
        column_names = [
            'Periodo de Cobertura',
            'Ingresos en Salud',
            'Gasto en Salud',
            'Porcentaje (%) de Siniestralidad',
        ]

    df_header = pd.read_excel(datapath, header=header_, nrows=2, usecols=columns_)
    df = pd.read_excel(datapath, header=None, skiprows=skip, nrows=rows_, usecols=columns_)
    
    df.columns = column_names
    
    Normalizar_fecha(df)
    df.to_csv(archivocsv, index=False)
   # return df

def Extraer_SiniestralidadFlujos(datapath, header_=10, rows_=23-7, skip=[], columns_="A:E", column_names=[], archivocsv='data.csv'):
    if not column_names:
        column_names = [
            'Ano de Cobertura',
            'Ingresos ARS Total',
            'Ingresos ARS Autogestion',
            'Ingresos ARS Privada',
            'Ingresos ARS Publica',
            'Gastos ARS Total',
            'Gastos ARS Autogestion',
            'Gastos ARS Privada',
            'Gastos ARS Publica',
            'Siniestralidad ARS Total',
            'Siniestralidad ARS Autogestion',
            'Siniestralidad ARS Privada',
            'Siniestralidad ARS Publica',    
        ]
    if not skip:
        skip = [12,30,48]

    df_header = pd.read_excel(datapath, header=header_, nrows=2, usecols=columns_)
    df1 = pd.read_excel(datapath, header=None, skiprows=skip[0], nrows=rows_, usecols=columns_)
    df1.columns = column_names[:5]
    Normalizar_fecha(df1)
    
    df2 = pd.read_excel(datapath, header=None, skiprows=skip[1], nrows=rows_, usecols=columns_)
    df2.columns = [column_names[0]] + column_names[5:9]
    Normalizar_fecha(df2)
    
    df3 = pd.read_excel(datapath, header=None, skiprows=skip[2], nrows=rows_, usecols=columns_)
    df3.columns = [column_names[0]] + column_names[9:]
    Normalizar_fecha(df3)
    df = pd.merge(df1, df2, on='Ano de Cobertura', how='inner')
    df = pd.merge(df, df3, on='Ano de Cobertura', how='inner')
    df.columns = column_names
    
    df.to_csv(archivocsv, index=False)
    #return df
    
def extraer_afiliados():
    """
    Extrae y normaliza los datos de afiliados.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
    data_path="data/SISALRIL/afiliacion/Afiliacion_RC_PBS_02.xlsx"
    datafile ='data/SISALRIL/afiliacion/RC_Afiliados_Edad_Sexo.csv'
    Extraer_Afiliados_Edad_Sexo(data_path,  archivocsv=datafile)
    
    data_path = "data/SISALRIL/afiliacion/Afiliacion_RC_PBS_03.xlsx"
    datafile = 'data/SISALRIL/afiliacion/RC_TasaDependencia_RC_ARS.csv'
    Extraer_TasaDependencia_RC_ARS(data_path,  archivocsv=datafile)
    
    data_path = "data/SISALRIL/afiliacion/Afiliacion_RC_PBS_07.xlsx"
    datafile= 'data/SISALRIL/afiliacion/RC_AfiliadosCotizantes_Edad_Sexo.csv'
    Extraer_AfiliadosCotizantes_Edad_Sexo(data_path,  archivocsv=datafile)
    
    data_path = "data/SISALRIL/afiliacion/Afiliacion_RC_PBS_08.xlsx"
    datafile= 'data/SISALRIL/afiliacion/RC_AfiliadosNoCotizantes_Edad_Sexo.csv'
    Extraer_AfiliadosNoCotizantes_Edad_Sexo(data_path,  archivocsv=datafile)
    
    data_path = "data/SISALRIL/afiliacion/Afiliacion_RC_PBS_04.xlsx"
    datafile= 'data/SISALRIL/afiliacion/RC_Tabla_Regiones.csv' 
    Extraer_Tabla_Regiones(data_path,  archivocsv=datafile)
  
    data_path = "data/SISALRIL/afiliacion/Afiliacion_RC_PBS_04.xlsx"
    datafile ='data/SISALRIL/afiliacion/RC_Datos_Regionales.csv'
    Extraer_datos_regionales(data_path, header_=7, rows_=54-7, archivocsv=datafile)
    
    data_path = "data/SISALRIL/afiliacion/Afiliacion_RC_PBS_05.xlsx"
    datafile ='data/SISALRIL/afiliacion/RC_Datos_Regionales_Cotizantes.csv'
    Extraer_datos_regionales(data_path, header_=7, rows_=54-7, archivocsv=datafile)
    
    data_path = "data/SISALRIL/afiliacion/Afiliacion_RC_PBS_06.xlsx"
    datafile ='data/SISALRIL/afiliacion/RC_Datos_Regionales_NoCotizantes.csv'
    Extraer_datos_regionales(data_path, header_=6, rows_=54-7, archivocsv=datafile)
    
    data_path = "data/SISALRIL/afiliacion/Afiliacion_SFS_PBS_03.xlsx"
    datafile = 'data/SISALRIL/afiliacion/SFS_Afiliacion_Tasa_Regimen.csv'
    Extraer_SFS_Afiliacion_Tasa_Regimen(data_path,  archivocsv=datafile)
    
    data_path = "data/SISALRIL/afiliacion/Afiliacion_SFS_PBS_05.xlsx"
    datafile = 'data/SISALRIL/afiliacion/SFS_Regimen_Sexo.csv'
    Extraer_SFS_Regimen_Sexo(data_path,  archivocsv=datafile)
    
    data_path = "data/SISALRIL/afiliacion/Afiliacion_SFS_PBS_07.xlsx"
    datafile = 'data/SISALRIL/afiliacion/SFS_Porcentaje_Regimen.csv'
    Extraer_SFS_Porcentaje_Regimen(data_path,  archivocsv=datafile)

def extraer_financiamiento():
    """
    Extrae y normaliza los datos de financiamiento.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
    data_path = "data/SISALRIL/financiamiento/Financiamiento_RC_PBS_01.xlsx"
    datafile = 'data/SISALRIL/financiamiento/CapitasDispersadoARS_TipoAfiliado_postMes.csv'
    Extraer_FinanciamientoDispersado_TipoAfiliado(data_path, header_=7, rows_=204-8, skip=8, columns_="A:M", column_names=[], archivocsv=datafile)
    
    data_path = "data/SISALRIL/financiamiento/Financiamiento_RC_PBS_01.xlsx"
    datafile = 'data/SISALRIL/financiamiento/MontoDispersadoARS_TipoAfiliado_postMes.csv'
    Extraer_FinanciamientoDispersado_TipoAfiliado(data_path, header_=7, rows_=401-205, skip=205, columns_="A:M", column_names=[], archivocsv=datafile)

    data_path = "data/SISALRIL/financiamiento/Financiamiento_RC_PBS_02.xlsx"
    datafile = 'data/SISALRIL/financiamiento/MontoDispersado_TipoAfiliado_Periodo.csv'
    names_ = [
                'Periodo de Cobertura',
                'Total_Capitas_Dispersadas',
                'Total_Capitas_Titulares',
                'Total_Capitas_Depend_Directos',
                'Total_Capitas_Depend_Adicionales',
                'Monto_Dispersado_Total',
                'Monto_Dispersado_Titulares',
                'Monto_Dispersado_Dep_Directos',
                'Monto_Dispersado_Dep_Adicionales',

            ]
    Extraer_FinanciamientoDispersado_TipoAfiliado(data_path, header_=6, rows_=202-6, skip=6, columns_="A:I", column_names=names_, archivocsv=datafile)


    data_path = "data/SISALRIL/financiamiento/Financiamiento_RC_PBS_03.xlsx"
    datafile = 'data/SISALRIL/financiamiento/CapitasDispersadoARS_Autogestion_postMes.csv'
    Extraer_FinanciamientoDispersado_TipoAfiliado(data_path, header_=7, rows_=204-8, skip=8, columns_="A:M", column_names=[], archivocsv=datafile)
    
    data_path = "data/SISALRIL/financiamiento/Financiamiento_RC_PBS_03.xlsx"
    datafile = 'data/SISALRIL/financiamiento/MontoDispersadoARS_Autogestion_postMes.csv'
    Extraer_FinanciamientoDispersado_TipoAfiliado(data_path, header_=7, rows_=401-205, skip=205, columns_="A:M", column_names=[], archivocsv=datafile)

    data_path = "data/SISALRIL/financiamiento/Financiamiento_RC_PBS_04.xlsx"
    datafile = 'data/SISALRIL/financiamiento/CapitasDispersadoARS_Privada_postMes.csv'
    Extraer_FinanciamientoDispersado_TipoAfiliado(data_path, header_=7, rows_=204-8, skip=8, columns_="A:M", column_names=[], archivocsv=datafile)
    
    data_path = "data/SISALRIL/financiamiento/Financiamiento_RC_PBS_04.xlsx"
    datafile = 'data/SISALRIL/financiamiento/MontoDispersadoARS_Privada_postMes.csv'
    Extraer_FinanciamientoDispersado_TipoAfiliado(data_path, header_=7, rows_=401-205, skip=205, columns_="A:M", column_names=[], archivocsv=datafile)

    data_path = "data/SISALRIL/financiamiento/Financiamiento_RC_PBS_05.xlsx"
    datafile = 'data/SISALRIL/financiamiento/CapitasDispersadoARS_Publicas_postMes.csv'
    Extraer_FinanciamientoDispersado_TipoAfiliado(data_path, header_=7, rows_=204-8, skip=8, columns_="A:M", column_names=[], archivocsv=datafile)
    
    data_path = "data/SISALRIL/financiamiento/Financiamiento_RC_PBS_05.xlsx"
    datafile = 'data/SISALRIL/financiamiento/MontoDispersadoARS_Publicas_postMes.csv'
    Extraer_FinanciamientoDispersado_TipoAfiliado(data_path, header_=7, rows_=401-205, skip=205, columns_="A:M", column_names=[], archivocsv=datafile)

    names_ = [
                    'Periodo de Cobertura',
                    'Total Empresas con aportes',
                    'Recaudo SFS',
                    'Cuidado de la Salud',
                    'Estancias Infantiles',
                    'Subsidios',
                    'Comisión Operación SISALRIL',
                    'Cápita Adicional',
                    'Recargo por Atraso en pago de Facturas',

                ]
    data_path = "data/SISALRIL/financiamiento/Financiamiento_RC_SFS_01.xlsx"
    datafile = 'data/SISALRIL/financiamiento/MontoSFS_Periodo_Cuenta.csv'
    Extraer_FinanciamientoDispersado_TipoAfiliado(data_path, header_=10, rows_=208-11, skip=11, columns_="A:I", column_names=names_, archivocsv=datafile)
    
    names_ = [
                    'Periodo de Cobertura',
                    'Salario Mínimo Cotizable',
                    'Tope de Salario Mínimo Cotizable',
                ]
    data_path = "data/SISALRIL/financiamiento/Financiamiento_SFS_03.xlsx"
    datafile = 'data/SISALRIL/financiamiento/SFS_TopeSalarioMinimoContizable.csv'
    Extraer_FinanciamientoDispersado_TipoAfiliado(data_path, header_=8, rows_=255-9, skip=9, columns_="A:C", column_names=names_, archivocsv=datafile)
    
    names_ =  [
                'Ano de Cobertura',    
                'Total_Salud (A)',
                'PDSS_Subsidiado (A)',
                'PDSS_Contributivo (A)',
                'Otros_Planes_de_Salud (A)',
                'PIB/Precios Corrientes (B)',
                'Total en Relación al PIB (A/B)',
                'PDSS_Subsidiado en Relación al PIB (A/B)',
                'PDSS_Contributivo en Relación al PIB (A/B)',
                'Otros_Planes_de_Salud en Relación al PIB (A/B)'
            ]
    data_path = "data/SISALRIL/financiamiento/Financiamiento_SFS_05.xlsx"
    datafile = 'data/SISALRIL/financiamiento/GastoSalud_PIB.csv'
    Extraer_FinanciamientoAnual(data_path, header_=9, rows_=23-9, skip=9, columns_="A:J", column_names=names_, archivocsv=datafile)
    
    names_ =  [
                'Ano de Cobertura',    
                'Total Dispersado SFS',
                'Régimen Contributivo',
                'Régimen Subsidiado',
            ]
    data_path = "data/SISALRIL/financiamiento/Financiamiento_SFS_PBS_01.xlsx"
    datafile = 'data/SISALRIL/financiamiento/MontoARS_PBS_Regimen.csv'
    Extraer_FinanciamientoAnual(data_path, header_=9, rows_=26-9, skip=9, columns_="A:D", column_names=names_, archivocsv=datafile)
    
    
def extraer_prestaciones():
    """
    Extrae y normaliza los datos de prestaciones.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
    names_ =  [
            'Ano de Cobertura',
            'Grupo Numero',
            'Grupo Descripcion ',
            'Servicios Prestados',
            'Distribucion Porcentual ',
        ]
    data_path = "data/SISALRIL/prestaciones/Prestaciones_RC_PBS_02.xlsx"
    datafile = 'data/SISALRIL/prestaciones/PrestacionesPBS_Servicios.csv'
    Extraer_PrestacionesPBS(data_path, header_=7, rows_=231-7, skip=7, columns_="A:E", column_names=names_, archivocsv=datafile)
    
    data_path = "data/SISALRIL/prestaciones/Prestaciones_RC_PBS_01.xlsx"
    datafile = 'data/SISALRIL/prestaciones/PrestacionesPBS_Monto.csv'
    Extraer_PrestacionesPBS(data_path, header_=6, rows_=230-6, skip=6, columns_="A:E",  archivocsv=datafile)

def extraer_siniestralidad():
    """
    Extrae y normaliza los datos de siniestralidad.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    """
    data_path = "data/SISALRIL/siniestralidad/Siniestralidad_RC_PBS_01.xlsx"
    datafile = 'data/SISALRIL/siniestralidad/RC_Ingresos_Gastos_Siniestralidad.csv'
    Extraer_SiniestralidadPBS(data_path, header_=9, rows_=204-9, skip=9, columns_="A:D",  archivocsv=datafile)
    
    data_path = "data/SISALRIL/siniestralidad/Sinestralidad_OP_01.xlsx"
    datafile = 'data/SISALRIL/siniestralidad/OP_Ingresos_Gastos_Siniestralidad.csv'
    Extraer_SiniestralidadPBS(data_path, header_=9, rows_=204-9, skip=9, columns_="A:D",  archivocsv=datafile)

    data_path = "data/SISALRIL/siniestralidad/Siniestralidad_RC_PBS_02.xlsx"
    datafile = 'data/SISALRIL/siniestralidad/RC_Ingresos_Gastos_Siniestralidad_Anual.csv'
    Extraer_SiniestralidadFlujos(data_path, header_=10, rows_=23-6, skip=[], columns_="A:E", archivocsv=datafile)
    
    data_path = "data/SISALRIL/siniestralidad/Sinestralidad_OP_02.xlsx"
    datafile = 'data/SISALRIL/siniestralidad/OP_Ingresos_Gastos_Siniestralidad_Anual.csv'
    Extraer_SiniestralidadFlujos(data_path, header_=14, rows_=31-14, skip=[14,32,50], columns_="A:E", archivocsv=datafile)

    
def merge_and_save_dataframes(df1, df2, output_file, on_columns, suffixes):
    # Realizar la unión de los DataFrames en base a columnas y agregar sufijos
    df_union = pd.merge(df1, df2, on=on_columns, suffixes=suffixes)
    
    # Imprimir información sobre el DataFrame resultante
    print(df_union.info())
    
    # Guardar el resultado en un archivo CSV
    df_union.to_csv(output_file, index=False)
    
def merge_and_save_dataframes(dfs, output_file, on_columns, suffixes):
    # Verificar que haya al menos dos DataFrames para unir
    if len(dfs) < 2:
        raise ValueError("Se requieren al menos dos DataFrames para unir.")
    
    # Realizar la unión de los DataFrames en base a columnas y agregar sufijos
    df_union = pd.merge(dfs[0], dfs[1], on=on_columns, suffixes=suffixes)
    
    # Unir los DataFrames restantes
    for df in dfs[2:]:
        df_union = pd.merge(df_union, df, on=on_columns)
    
    # Imprimir información sobre el DataFrame resultante
    print(df_union.info())
    
    # Guardar el resultado en un archivo CSV
    df_union.to_csv(output_file, index=False)

    
    

@st.cache_data
def load_dataframes():
  """
  Carga todos los Dataframes en una sesión de Streamlit y devuelve un diccionario con los nombres y Dataframes.

  Parámetros:
    Ninguno.

  Retorno:
    Diccionario con los nombres y Dataframes.
  """
  # Lista de archivos
  Tablas = [
    'data/Base_Creada/Afiliados_Edad_Sexo_Cotizacion.csv',
    'data/Base_Creada/RC_Tabla_Regiones.csv',
    'data/Base_Creada/RC_Datos_Regionales_Cotizacion.csv',
    'data/Base_Creada/RC_Region_Salud_Total.csv',
    'data/Base_Creada/RC_Region_Geografica_Total_Combinado.csv',
    'data/Base_Creada/SFS_Regimen_Sexo_Porcentaje_Tasa.csv',
    'data/Base_Creada/Financiamiento_Dispersado_TipoARS_postMes.csv',
    'data/Base_Creada/Financiamiento_Dispersado_Salario.csv',
    'data/Base_Creada/FinanciamientoARS_SaludPIB_Regimen.csv',
    'data/Base_Creada/PrestacionesPBS.csv',
    'data/Base_Creada/Ingresos_Gastos_Siniestralidad.csv',
    'data/Base_Creada/Ingresos_Gastos_Siniestralidad_Anual.csv'
  ]

  # Lista de nombres específicos
  Nombres_Especificos = [
    'Afiliados Edad Sexo Cotizacion',
    'Tabla Regiones',
    'Datos Regionales Cotizacion',
    'Region Salud Total',
    'Region Geografica Total Combinado',
    'SFS Regimen Sexo Porcentaje Tasa',
    'Financiamiento Dispersado TipoARS postMes',
    'Financiamiento Dispersado Salario',
    'FinanciamientoARS SaludPIB Regimen',
    'PrestacionesPBS',
    'Ingresos Gastos Siniestralidad',
    'Ingresos Gastos Siniestralidad Anual'
  ]

  # Diccionario para almacenar Dataframes
  dataframes = {}

  # Cargar Dataframes
  for i in range(len(Tablas)):
    dataframes[Nombres_Especificos[i]] = pd.read_csv(Tablas[i])
  # Guardar Dataframes y nombres específicos en Session State
    column_descriptions = {
        'Afiliados Edad Sexo Cotizacion': {
            'Grupo Numero': 'Número de grupo de afiliados',
            'Grupo Descripcion': 'Descripción del grupo de afiliados',
            'Sexo': 'Sexo de los afiliados',
            'Rango de Edad': 'Rango de edad de los afiliados',
            'Cantidad Afiliados': 'Cantidad de afiliados en el grupo'
        },
        'Tabla Regiones': {
            'Región Geográfica/2': 'Región geográfica',
            'Región Salud': 'Región de salud',
            'Provincia': 'Provincia'
        },
        'Datos Regionales Cotizacion': {
              "Total general": "Número total de afiliados en la región.",
              "Total Región Distrito Nacional a San Juan de la Maguana": "Número total de afiliados en cada provincia o distrito.",
              "No Especificada": "Número de afiliados cuya ubicación no está especificada.",
              "Year": "Año al que se refiere el registro.",
              "Month": "Mes al que se refiere el registro.",
              "Periodo de Cobertura": "Periodo de tiempo al que se refiere el registro.",
              "Cotizante": "Tipo de afiliado (cotizante o no cotizante)."

        },
        'Region Salud Total': {
          "Periodo de Cobertura": "Periodo de tiempo al que se refiere el registro.",
          "Year": "Año al que se refiere el registro.",
          "Month": "Mes al que se refiere el registro.",
          "Total general a Total VI - El Valle": "Número total de afiliados en cada región de salud.",
          "Cotizante": "Tipo de afiliado (cotizante o no cotizante)."
         },
        'Region Geografica Total Combinado': {
          "Periodo de Cobertura": "Periodo de tiempo al que se refiere el registro.",
          "Year": "Año al que se refiere el registro.",
          "Month": "Mes al que se refiere el registro.",
          "Total general a Total Región Sur": "Número total de afiliados en cada región geográfica.",
          "Cotizante": "Tipo de afiliado (cotizante o no cotizante)."
         },
        'SFS Regimen Sexo Porcentaje Tasa': {
           "Periodo de Cobertura": "Periodo de tiempo al que se refiere el registro.",
          "SFS_Total, SFS_Hombres, SFS_Mujeres": "Número total de afiliados en el Seguro Familiar de Salud (SFS) según sexo.",
          "Reg_Subsidiado_Total, Reg_Subsidiado_Hombres, Reg_Subsidiado_Mujeres": "Número total de afiliados en el régimen subsidiado según sexo.",
          "RC_Total, RC_Hombres, RC_Mujeres": "Número total de afiliados en el régimen contributivo según sexo.",
          "Year": "Año al que se refiere el registro.",
          "Month": "Mes al que se refiere el registro.",
          "Poblacion Total proyectada": "Población total proyectada.",
          "Porcentaje de Población Cubierta por el SFS": "Porcentaje de la población cubierta por el SFS.",
          "Total Seguro Familiar de Salud": "Número total de personas cubiertas por el Seguro Familiar de Salud.",
          "Reg_Subsidiado, Reg_Contributivo": "Número total de personas en el régimen subsidiado y contributivo, respectivamente.",
          "Tasa_Dep_Regimen_Subsidiado, RC_Tasa_Dependencia, RC_Tasa_Dependencia_Directa": "Tasas de dependencia de los diferentes regímenes."
        },
        'Financiamiento Dispersado TipoARS postMes': {
          "Periodo de Cobertura": "Periodo de tiempo al que se refiere el registro.",
          "Total_Capitas_Dispersadas, Total_Capitas_Dispersada_mes, Total_Capitas_Dispersada_posterior": "Total de capitas dispersadas en total, en el mes actual y en meses posteriores, respectivamente.",
          "Titulares_Total_capitas, Titulares_Dispersadas_mes_capitas, Titulares_Dispersadas_posterior_capitas": "Número total de titulares de capitas, dispersados en el mes actual y en meses posteriores, respectivamente.",
          "Dependientes_Total_capitas, Dependientes_Dispersadas_mes_capitas, Dependientess_Dispersadas_posterior_capitas": "Número total de dependientes de capitas, dispersados en el mes actual y en meses posteriores, respectivamente.",
          "Adicionales_Total_capitas, Adicionales_Dispersadas_mes_capitas, Adicionales_Dispersadas_posterior_capitas": "Número total de adicionales de capitas, dispersados en el mes actual y en meses posteriores, respectivamente.",
          "Year, Month": "Año y mes al que se refiere el registro.",
          "Total_Monto_Dispersadas, Total_Monto_Dispersada_mes, Total_Monto_Dispersada_posterior": "Monto total dispersado en total, en el mes actual y en meses posteriores, respectivamente.",
          "Tipo_de_ARS": "Tipo de Administradora de Riesgos de Salud (ARS)."
        },
        'Financiamiento Dispersado Salario': {
          "Periodo de Cobertura": "Periodo de tiempo al que se refiere el registro.",
          "Total Empresas con aportes": "Número total de empresas con aportes.",
          "Recaudo SFS, Cuidado de la Salud, Estancias Infantiles, Subsidios, Comisión Operación SISALRIL, Cápita Adicional, Recargo por Atraso en pago de Facturas": "Montos relacionados con el financiamiento.",
          "Year, Month": "Año y mes al que se refiere el registro.",
          "Salario Mínimo Cotizable, Tope de Salario Mínimo Cotizable": "Montos relacionados con el salario mínimo cotizable y su tope.",
          "Total_Monto_Dispersadas": "Monto total dispersado.",
          "Total_Capitas_Titulares, Total_Capitas_Depend_Directos, Total_Capitas_Depend_Adicionales": "Número total de capitas de titulares, dependientes directos y dependientes adicionales, respectivamente.",
          "Monto_Dispersado_Total, Monto_Dispersado_Titulares, Monto_Dispersado_Dep_Directos, Monto_Dispersado_Dep_Adicionales": "Montos dispersados totales y por tipo de afiliado."
         },
        'FinanciamientoARS SaludPIB Regimen': {
          "Ano de Cobertura": "Año al que se refiere el registro.",
          "Total_Salud (A), PDSS_Subsidiado (A), PDSS_Contributivo (A), Otros_Planes_de_Salud (A)": "Montos relacionados con la salud en diferentes regímenes.",
          "PIB/Precios Corrientes (B)": "Producto Interno Bruto (PIB) a precios corrientes.",
          "Total en Relación al PIB (A/B), PDSS_Subsidiado en Relación al PIB (A/B), PDSS_Contributivo en Relación al PIB (A/B), Otros_Planes_de_Salud en Relación al PIB (A/B)": "Relación de los montos de salud con el PIB.",
          "Total Dispersado SFS, Régimen Contributivo, Régimen Subsidiado": "Montos dispersados y relacionados con los diferentes regímenes."
        },
        'PrestacionesPBS': {
           "Ano de Cobertura": "Año al que se refiere el registro.",
          "Total_Salud (A), PDSS_Subsidiado (A), PDSS_Contributivo (A), Otros_Planes_de_Salud (A)": "Montos relacionados con la salud en diferentes regímenes.",
          "PIB/Precios Corrientes (B)": "Producto Interno Bruto (PIB) a precios corrientes.",
          "Total en Relación al PIB (A/B), PDSS_Subsidiado en Relación al PIB (A/B), PDSS_Contributivo en Relación al PIB (A/B), Otros_Planes_de_Salud en Relación al PIB (A/B)": "Relación de los montos de salud con el PIB.",
          "Total Dispersado SFS, Régimen Contributivo, Régimen Subsidiado": "Montos dispersados y relacionados con los diferentes regímenes."
       },
        'Ingresos Gastos Siniestralidad': {
          "Periodo de Cobertura": "Periodo de tiempo al que se refiere el registro.",
          "Ingresos en Salud_RC, Gasto en Salud_RC": "Montos de ingresos y gastos en salud para el Régimen Contributivo.",
          "Porcentaje (%) de Siniestralidad_RC": "Porcentaje de siniestralidad para el Régimen Contributivo.",
          "Ingresos en Salud_OP, Gasto en Salud_OP": "Montos de ingresos y gastos en salud para el Régimen de Pensionados y Jubilados.",
          "Porcentaje (%) de Siniestralidad_OP": "Porcentaje de siniestralidad para el Régimen de Pensionados y Jubilados.",
          "Year, Month": "Año y mes al que se refiere el registro."
      },
        'Ingresos Gastos Siniestralidad Anual': {
          "Ano de Cobertura": "Año al que se refiere el registro.",
          "Ingresos ARS Total_RC, Ingresos ARS Autogestion_RC, Ingresos ARS Privada_RC, Ingresos ARS Publica_RC": "Montos de ingresos de las ARS en diferentes categorías (Régimen Contributivo).",
          "Gastos ARS Total_RC, Gastos ARS Autogestion_RC, Gastos ARS Privada_RC, Gastos ARS Publica_RC": "Montos de gastos de las ARS en diferentes categorías (Régimen Contributivo).",
          "Siniestralidad ARS Total_RC, Siniestralidad ARS Autogestion_RC, Siniestralidad ARS Privada_RC, Siniestralidad ARS Publica_RC": "Siniestralidad de las ARS en diferentes categorías (Régimen Contributivo).",
          "Ingresos ARS Total_OP, Ingresos ARS Autogestion_OP, Ingresos ARS Privada_OP, Ingresos ARS Publica_OP": "Montos de ingresos de las ARS en diferentes categorías (Régimen de Pensionados y Jubilados).",
          "Gastos ARS Total_OP, Gastos ARS Autogestion_OP, Gastos ARS Privada_OP, Gastos ARS Publica_OP": "Montos de gastos de las ARS en diferentes categorías (Régimen de Pensionados y Jubilados).",
          "Siniestralidad ARS Total_OP, Siniestralidad ARS Autogestion_OP, Siniestralidad ARS Privada_OP, Siniestralidad ARS Publica_OP": "Siniestralidad de las ARS en diferentes categorías (Régimen de Pensionados y Jubilados)."
     }
    }
  st.session_state.dataframes = dataframes
  st.session_state.nombres_especificos = Nombres_Especificos
  st.session_state.column_descriptions = column_descriptions
  return [dataframes,Nombres_Especificos,column_descriptions]
    


def filter_string(df, column, selected_list):
    final = []
    df = df[df[column].notna()]
    for idx, row in df.iterrows():
        if row[column] in selected_list:
            final.append(row)
    res = pd.DataFrame(final)
    return res

def number_widget(df, column, ss_name, container=None):
    df = df[df[column].notna()]
    max_val = float(df[column].max())
    min_val = float(df[column].min())
    temp_input = container.slider(f"{column.title()}", min_val, max_val, (min_val, max_val), key=ss_name)
    all_widgets.append((ss_name, "number", column))

def number_widget_int(df, column, ss_name, container):
    df = df[df[column].notna()]
    max_val = int(df[column].max())
    min_val = int(df[column].min())
    temp_input = container.slider(f"{column.title()}", min_val, max_val, (min_val, max_val), key=ss_name)
    all_widgets.append((ss_name, "number", column))

def create_select(df, column, ss_name, multi=False, container=None):
    df = df[df[column].notna()]
    options = df[column].unique()
    options.sort()
    if multi==False:
        temp_input = container.selectbox(f"{column.title()}", options, key=ss_name)
        all_widgets.append((ss_name, "select", column))
    else:
        temp_input = container.multiselect(f"{column.title()}", options, key=ss_name)
        all_widgets.append((ss_name, "multiselect", column))

def text_widget(df, column, ss_name, container):
    temp_input = container.text_input(f"{column.title()}", key=ss_name)
    all_widgets.append((ss_name, "text", column))

def create_widgets(df, create_data={}, ignore_columns=[], container=None):
    """
    This function will create all the widgets from your Pandas DataFrame and return them.
    df => a Pandas DataFrame
    create_data => Optional dictionary whose keys are the Pandas DataFrame columns
        and whose values are the type of widget you wish to make.
        supported: - multiselect, select, text
    ignore_columns => columns to entirely ignore when creating the widgets.
    container => Streamlit container where widgets should be created (e.g., st.sidebar)
    """
    for column in ignore_columns:
        df = df.drop(column, axis=1)
    global all_widgets
    all_widgets = []
    for ctype, column in zip(df.dtypes, df.columns):
        if column in create_data:
            if create_data[column] == "text":
                text_widget(df, column, column.lower(), container)
            elif create_data[column] == "select":
                create_select(df, column, column.lower(), multi=False, container=container)
            elif create_data[column] == "multiselect":
                create_select(df, column, column.lower(), multi=True, container=container)
        else:
            if ctype == "float64":
                number_widget(df, column, column.lower(), container)
            elif ctype == "int64":
                number_widget_int(df, column, column.lower(), container)
            elif ctype == "object":
                if str(type(df[column].tolist()[0])) == "<class 'str'>":
                    text_widget(df, column, column.lower(), container)
    return all_widgets

def filter_df(df, all_widgets):
    """
    This function will take the input dataframe and all the widgets generated from
    Streamlit Pandas. It will then return a filtered DataFrame based on the changes
    to the input widgets.

    df => the original Pandas DataFrame
    all_widgets => the widgets created by the function create_widgets().
    """
    res = df
    for widget in all_widgets:
        ss_name, ctype, column = widget
        data = st.session_state[ss_name]
        if data:
            if ctype == "text":
                if data != "":
                    res = res.loc[res[column].str.contains(data)]
            elif ctype == "select":
                res = filter_string(res, column, data)
            elif ctype == "multiselect":
                res = filter_string(res, column, data)
            elif ctype == "number":
                min_val, max_val = data
                res = res.loc[(res[column] >= min_val) & (res[column] <= max_val)]
    return res
