import shap
import pandas as pd
from neuralforecast import NeuralForecast
import numpy as np




class SHAP_TS:
  # initilization of the SHAP KernelExplainer
  def __init__(self, data, model, input_size, horizon, frequency ):
    self.data = data

    self.input_size = input_size
    self.horizon = horizon
    self.frequency = frequency
    self.model = self.prepare_model(model)

    self.model.fit(self.data)
    self.reference_set = self.convert_reference_set(self.data)   # this is the converted data for the reference set
    self.explainer = shap.KernelExplainer(self.predict, self.reference_set.drop(columns = 'starting_date', inplace = False))


  def get_explainer(self):
    return self.explainer




  def convert_reference_set(self, long_df):
    # taking the first row to get the starting date
    start_idx = long_df.index[0]
    end_idx = long_df.index[-1]
    self.starting_date = long_df.loc[start_idx, 'ds']
    self.ending_date = long_df.loc[end_idx, 'ds']


    # rolling sul dataframe per avere finestre lunghe quanto il context length
    # in modo da avere un array con shape (n_windows, n_timestamps + 1 (starting_date))

    # rolling the long df to create windows of size input_size
    # each window will be an array with shape ( n_windows, n_timestamps + 1 (starting_date) )
    rows = list()
    long_df = long_df['y']
    for window in long_df.rolling(window = self.input_size):
      if window.size >= self.input_size:
        rows.append( window.values.reshape( self.input_size) )


    rows = np.asarray(rows)

    # naming the timestamp features
    feature_list = ['timestamp_'+ str(x) for x in range(1,self.input_size+1)]
    df_ts = pd.DataFrame(rows, columns=feature_list)

    # defining the starting date which will be the same for all rows
    df_ts['starting_date'] = self.starting_date

    return df_ts


  def prepare_model(self, model):
    nf = model
    nf.fit(self.data)
    return nf


  # this method is used to load the prediction function of the model,
  # given an input x, it returns the predictions.
  # if x is a single instance, it returns the predictions for that instance.
  # if x is an array with multiple instances (like for the reference set), 
  # it returns the predictions for each instance.
  def predict(self, x):
    print(x.shape)
    preds = list()
    if x.shape[0] > 1:    # if the array has more than one instance,

      # ogni riga diventera un long_df da passare a get_datasets.
      # il dset_test ottenuto da ciascuna riga sara l'input da passare al modello per ottenere la predizione della riga
      # for each row in the array, we create a long_df and then we pass it to the model to get the predictions
      for row, idx in zip(x, range(len(x))):
            print(idx)
            long_df = pd.DataFrame(row, columns = ['y'])
            long_df['unique_id'] = 'T1'

            # for ds column
            if idx >= 1:  # since i have only one starting datetime and I have overlapping time windows,
                            # I need to calculate a delta to add to each row after the first one to give it an adequate timestamps column
                delta = pd.Timedelta(4 * idx, 's')
                long_df['ds'] = pd.date_range(start = self.starting_date, periods = len(row), freq = self.frequency) + delta

            else:
                long_df['ds'] = pd.date_range(start = self.starting_date,periods = len(row), freq = self.frequency)


            y_hat =  self.model.predict(long_df)
            preds.append(y_hat['NHITS'].values)


      preds = np.stack(preds)
      return preds


    # if it is a single instance, we take the values of all timestamps
    else:
      long_df = pd.DataFrame(x[0], columns = ['y'])
      long_df['unique_id'] = 'T1'
      long_df['ds'] = pd.date_range(start = self.starting_date, periods = len(x[0]), freq = self.frequency)


      y_hat = self.model.predict(long_df)
      preds = y_hat['NHITS'].values.reshape(1, self.horizon)
      return preds


      return preds



  def get_shaply(self, instance):
    instance = self.convert_reference_set(instance)
    shap_values = self.explainer.shap_values(instance.drop(columns = 'starting_date', inplace = False))
    return shap_values