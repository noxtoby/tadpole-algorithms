import logging

from pathlib import Path

import pandas as pd
import scipy as sp
import numpy as np
from scipy import stats
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

import statsmodels.formula.api as smf
import statsmodels.api as sm
import importlib.resources as pkg_resources

from tqdm.auto import tqdm

from tadpole_algorithms.models.tadpole_model import TadpoleModel

from datetime import datetime
from dateutil.relativedelta import relativedelta

import pystan
import os

logger = logging.getLogger(__name__)


def bootstrap(model, train_df, y_df, test_df, n_bootstraps: int = 100, confidence=0.50) -> float:
    """Runs model `model` using different random sampled train & test splits.

    Returns:
        float: Confidence Interval delta for `confidence` level.
    """
    predictions = []
    for i in tqdm(range(0, n_bootstraps)):
        train_df_resampled, y_df_resampled = resample(train_df, y_df, random_state=i)
        model.fit(train_df_resampled, y_df_resampled)
        prediction = model.predict(test_df)[0]
        predictions.append(prediction)

    se = sp.stats.sem(predictions)  # Standard error
    h = se * sp.stats.t._ppf((1 + confidence) / 2., len(y_df))  # CI

    # m-h and m+h give confidence interval
    return h

def check_for_save_file(file_name,function=None):
    if os.path.isfile(file_name):
        print('check_for_save_file(): File detected ({0}) - you can load data.'.format(file_name))
        #ebm_save = sio.loadmat(file_name)
        return 1
    else:
        if function is None:
            print('No save file found')
        else:
            print('No save file found: you should call your function {0}'.format(function.__name__))
        return 0


class DEM(TadpoleModel):
    """Differential Equation Model method, Neil Oxtoby - neil@neiloxtoby.com

    The `train_df*` attributes contain training data optimized for each variable.

    The `y_train_df*` attributes contain the labels to be used for training by each model,
    thus corresponding to the matching `train_df` DataFrame.

    Attributes:
        diagnosis_model (Pipeline): Model for predicting 'diagnosis' variable
        adas_model (Pipeline): Model for predicting 'ADAS13' variable
        ventricles_model (Pipeline): Model for predicting 'ventricles' variable

        y_diagnosis (pandas.DataFrame): 'Diagnosis' labels
        train_df_diagnosis (pandas.DataFrame): Training data used for 'diagnosis' model.
    """

    def __init__(self, confidence_intervals=True):
        # self.diagnosis_model = Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('classifier', svm.SVC(kernel='rbf', C=0.5, gamma='auto', class_weight='balanced', probability=True)),
        # ])
        # self.adas_model = Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('classifier', svm.SVR(kernel='rbf', C=0.5, gamma='auto')),
        # ])
        # self.ventricles_model = Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('classifier', svm.SVR(kernel='rbf', C=0.5, gamma='auto')),
        # ])

        self.df_dem_fits = None

        self.y_diagnosis  = None
        self.y_adas       = None
        self.y_ventricles = None

        self.train_df_diagnosis  = None
        self.train_df_adas       = None
        self.train_df_ventricles = None

        self.confidence_intervals = confidence_intervals

    @staticmethod
    def preprocess(train_df: pd.DataFrame):
        logger.info("Pre-processing")
        train_df = train_df.copy()
        if 'Diagnosis' not in train_df.columns:
            """We want to transform 'DXCHANGE' (a change in diagnosis, in contrast
            to the previous visits diagnosis) to an actual diagnosis."""
            train_df = train_df.replace({
               'DX': {
                   'NL'             :1,
                   'MCI to NL'      :1,
                   'Dementia to NL' :1,
                   'MCI'            :2,
                   'NL to MCI'      :2,
                   'Dementia to MCI':2,
                   'Dementia'       :3,
                   'MCI to Dementia':3,
                   'NL to Dementia' :3
               }
            })
            train_df.rename(columns={"DX": "Diagnosis"},inplace=True)
            #train_df['Diagnosis'] = train_df['DXCHANGE'].values
        
        #* Adjust AGE at baseline to VISITAGE
        train_df['VISITAGE'] = train_df['AGE'] + train_df['Years_bl']

        # Divide certain volumes by ICV. Could add more...
        dem_markers = [
            'WholeBrain_ICV', 'Hippocampus_ICV', 'Ventricles_ICV', 'Entorhinal_ICV',
            'MMSE', 'ADAS13'
        ]
        #* Avoid divide-by-zero errors: code by Esther Bron
        icv_bl_median = train_df['ICV_bl'].median()
        train_df.loc[train_df['ICV_bl'] == 0, 'ICV_bl'] = icv_bl_median

        for vol in ['WholeBrain', 'Hippocampus', 'Ventricles', 'Entorhinal']:
            train_df[vol+'_ICV'] = train_df[vol].values/train_df['ICV_bl'].values
            #test_df[vol+'_ICV'] = test_df[vol].values/test_df['ICV_bl'].values

        logger.info('Forcing Numeric Values')
        for i in range(5, len(dem_markers)):
            if train_df[dem_markers[i]].dtype != 'float64':
                train_df[dem_markers[i]] = pd.to_numeric(train_df[dem_markers[i]], errors='coerce')

        """Sort the DataFrame per patient on age (at time of visit). This allows using observations from
        the next row/visit to be used as a label for the previous row. (See `get_futures` method.)"""
        train_df = train_df.sort_values(by=['RID', 'VISITAGE'])

        #train_df = train_df.drop(['EXAMDATE', 'AGE', 'PTGENDER', 'PTEDUCAT', 'APOE4'], axis=1)
        """Select features based on Billabong_features.csv file"""
        features_csv_path = Path(__file__).parent / 'Billabong_features.csv'
        selected_features = pd.read_csv(features_csv_path)['feature'].values.tolist()
        selected_features += ['RID', 'Diagnosis', 'Years_bl']
        selected_features = set(selected_features)
        train_df = train_df.copy()[selected_features]

        train_df = DEM.fill_nans_by_older_values(train_df)

        #* Preprocessing for DEM model
        df_dem = DEM.dem_gradients(
            df=train_df,
            markers=dem_markers,
            t_col='Years_bl',
            dx_col='Diagnosis'
        )
        #* Postselect: exclude non-progressing, non-abnormal
        df_dem_, df_postselection = DEM.dem_postselect(df_dem=df_dem,markers=dem_markers,dx_col='Diagnosis.bl')

        #* Optionally subsample to reduce the dimensions (fitting can take quite some time)
        subsample = 3
        #- Sort biomarkers to approximate getting even coverage
        df_dem__ = df_dem_.sort_values(by=[d+'-mean' for d in dem_markers])
        df_dem__.reset_index(drop=True, inplace=True)
        n_subsample = int(df_dem__.shape[0]/subsample)
        s = np.arange(0,df_dem__.shape[0],int(df_dem__.shape[0]/n_subsample))
        df_dem__ = df_dem__.loc[s]

        return df_dem__

    @staticmethod
    def dxdt(x,t):
        #* Fit a GLM using statsmodels
        glm_formula = 'x ~ t'
        mod = smf.ols(formula=glm_formula, data={'x':x,'t':t})
        res = mod.fit()
        return res.params[1]

    @staticmethod
    def dem_gradients(df,
            markers,
            id_col='RID',
            t_col='Years_bl',
            dx_col = 'Diagnosis',
            n_timepoints_min=2):
        """
        dem_gradients()
        Calculates individual gradients from longitudinal data and 
        returns a cross-section of differential data
        Neil Oxtoby, UCL, November 2018
        """
        #* Remove individuals without enough data
        counts = df.groupby([id_col]).agg(['count'])
        counts.reset_index(inplace=True)
        has_long_data = (np.all(counts>=n_timepoints_min,axis=1))
        rid_include = counts[id_col][ has_long_data ].values
        #* Add baseline DX
        counts = counts.merge(df.loc[df[t_col]==0,[id_col,dx_col]].rename(columns={dx_col:dx_col+'.bl'}),on='RID')
        dxbl_include = counts[dx_col+'.bl'][ has_long_data ].values
        #* Baseline DX
        df = df.merge(df.loc[df[t_col]==0,[id_col,dx_col]].rename(columns={dx_col:dx_col+'.bl'}))
        id_dxbl = df[[id_col,dx_col+'.bl']]
        #* Keep only RID included
        df_ = df.loc[ df[id_col].isin(rid_include) ]
        #* Add baseline DX
        df_ = df_.merge(id_dxbl)

        #* Calculate gradients
        df_dem = pd.DataFrame(data={id_col:rid_include,dx_col+'.bl':dxbl_include})
        for i in df_dem[id_col]:
            rowz = i==df_[id_col]
            rowz_dem = i==df_dem[id_col]
            t = df_.loc[rowz,t_col]
            for m in markers:
                x = df_.loc[rowz,m]
                df_dem.loc[rowz_dem,m+'-mean'] = np.mean(x)
                df_dem.loc[rowz_dem,m+'-grad'] = DEM.dxdt(x,t)

        return df_dem

    @staticmethod
    def dem_postselect(df_dem,markers,dx_col='Diagnosis'):
        """
        Postselects differential data, a la Villemagne 2013:
        - Omits non-progressing (negative gradient), non-abnormal (less than biomarker median of CN) differential data
        Neil Oxtoby, UCL, November 2018
        """
        #dx_dict = {1:'CN',2:'MCI',3:'AD',4:'CNtoMCI',5:'MCItoAD',6:'CNtoAD',7:'MCItoCN',8:'ADtoMCI',9:'ADtoCN'}
        x_text = '-mean'
        y_text = '-grad'

        df_postelection = pd.DataFrame(data={'Marker':markers})

        #* 1. Restrict to MCI and AD - purifies, but might also remove presymptomatics in CN
        dx_included = [2,3]
        df_ = df_dem.loc[df_dem[dx_col].isin(dx_included)].copy()

        #* 2. Exclude normal and non-progressing
        for m in markers:
            #* 2.1 Normal threshold = median of CN (alt: use clustering)
            normal_threshold = df_dem.loc[df_dem[dx_col].isin([1]),m+x_text].median()
            #* 2.2 Non-progressing = negative gradient
            nonprogress_threshold = 0
            excluded_rows = (df_[m+x_text] < normal_threshold) & (df_[m+y_text] < nonprogress_threshold)

            df_postelection.loc[df_postelection['Marker']==m,'Normal-Threshold'] = normal_threshold

        return df_, df_postelection

    @staticmethod
    def fit_dem(df_dem,stan_model):
        """
        dem_fit = fit_dem(df_dem,stan_model)
        """
        x_text = '-mean'
        y_text = '-grad'
        markers = df_dem.columns.tolist()
        markers = [m.replace(x_text,'') for m in markers if 'mean' in m]

        df_dem_fits = pd.DataFrame(data={'Marker':markers})

        # #* 1. Linear regression
        # slope, intercept, r_value, p_value, std_err = stats.linregress(x_,dxdt_)
        # DEMfit = {'linreg_slope':slope}
        # DEMfit['linreg_intercept'] = intercept
        # DEMfit['linreg_r_value'] = r_value
        # DEMfit['linreg_p_value'] = p_value
        # DEMfit['linreg_std_err'] = std_err

        for m in markers:
            x = df_dem[m+x_text].values
            y = df_dem[m+y_text].values
            i = np.argsort(x)
            x = x[i]
            y = y[i]

            #* GPR setup: hyperparameters, etc.
            x_scale = (max(x)-min(x))
            y_scale = (max(y)-min(y))
            sigma_scale = 0.1*y_scale

            x_predict = np.linspace(min(x),max(x),20)
            N_predict = len(x_predict)
            #* MCMC CHAINS: initial values
            rho_i = x_scale/2
            alpha_i = y_scale/2
            sigma_i = sigma_scale
            init = {'rho':rho_i, 'alpha':alpha_i, 'sigma':sigma_i}
            dem_gpr_dat = {
                'N': len(x),
                'x': x,
                'y': y,
                'x_scale' : x_scale,
                'y_scale' : y_scale,
                'sigma_scale' : sigma_scale,
                'x_predict'   : x_predict,
                'N_predict'   : N_predict
            }
            df_dem_fits.loc[df_dem_fits['Marker']==m,'x_predict'] = x_predict

            txt = 'Performing GPR for {0}'.format(m)
            print(txt)
            logger.info(txt)
            fit = stan_model.sampling(data=dem_gpr_dat,
                                      init=[init,init,init,init],
                                      iter=1000,
                                      chains=4)
            df_dem_fits.loc[df_dem_fits['Marker']==m,'pystan_fit_gpr'] = fit
        return df_dem_fits

    @staticmethod
    def get_futures(train_df, features=['RID', 'Diagnosis', 'ADAS13', 'Ventricles_ICV']):
        """For each feature in `features` argument, generate a `Future_{feature}` column, that is filled
        using the next row for each patient"""

        futures_df = train_df[features].copy()

        # Get future value from each row's next row, e.g. shift the column one up
        for predictor in ["Diagnosis", "ADAS13", 'Ventricles_ICV']:
            futures_df["Future_" + predictor] = futures_df[predictor].shift(-1)

        # Drop each last row per patient
        futures_df = futures_df.drop(futures_df.groupby('RID').tail(1).index.values)
        return futures_df

    @staticmethod
    def fill_nans_by_older_values(train_df):
        """Fill nans in feature matrix by older values (ffill), then by newer (bfill)"""

        df_filled_nans = train_df.groupby('RID').fillna(method='ffill').fillna(method='bfill')
        train_df[df_filled_nans.columns] = df_filled_nans
        return train_df

    def train(self, train_df):
        df_dem = self.preprocess(train_df)

        #* Esther code
        # futures = self.get_futures(train_df)
        # # Not part of `preprocess` because it's needed for the futures.
        # train_df = train_df.drop(['RID'], axis=1)
        # # Fill left over nans with mean
        # train_df = train_df.fillna(train_df.mean())
        # train_df = train_df.fillna(0)
        # def non_nan_y(_train_df, _y_df):
        #     """Drops all rows with a `y` value that is NaN
        #
        #     Returns:
        #         Tuple containing (`train_df`, `y_df`), without NaNs for `y_df`.
        #     """
        #
        #     # indices where the y value is not nan
        #     not_nan_idx = _y_df[_y_df.notna()].index
        #
        #     # return from both the train dataframe and y the records with these indices
        #     return _train_df.loc[not_nan_idx], _y_df[not_nan_idx]
        #
        # self.train_df_diagnosis, self.y_diagnosis = non_nan_y(train_df, futures['Future_Diagnosis'])
        # self.train_df_adas, self.y_adas = non_nan_y(train_df, futures['Future_ADAS13'])
        # self.train_df_ventricles, self.y_ventricles = non_nan_y(train_df, futures['Future_Ventricles_ICV'])

        logger.info("Training models")
        # Prep for Gaussian Process regression
        gpr_stan_code = Path(__file__).parent / 'gp_betancourt.stan'
        dem_gpr_stan_model = pystan.StanModel(file=gpr_stan_code.as_posix(),model_name='gpr_dem')
        # Check for saved results and fit
        fname_save_pystan_fits = "DEM-results-pickle-pystan-fits"
        check_flag = check_for_save_file(file_name=fname_save_pystan_fits,function=None)
        if check_flag: #"dem_gpr_stan_model_fits" in pystan_results:
            print('DEM.train():           Not fitting DEMs: existing results detected.')
            pickle_file = open(fname_save_pystan_fits,'rb')
            pystan_results = pickle.load(pickle_file)
            dem_gpr_stan_model_fits = pystan_results["dem_gpr_stan_model_fits"]
            pickle_file.close()
            #dem_gpr_stan_model_fits = pystan_results["dem_gpr_stan_model_fits"]
        else:
            print('DEM.train():           Fitting DEMs using GP regression')
            self.df_dem_fits = self.fit_dem(df_dem,dem_gpr_stan_model)
            #* Save the fits to a new pickle file - the StanModel mmust be unpickled first
            pystan_results["dem_gpr_stan_model_fits"] = df_dem_fits
            pickle_file = open(fname_save_pystan_fits,'wb')
            pickle_output = pickle.dump(pystan_results,pickle_file)
            pickle_file.close()

        # self.diagnosis_model.fit(self.train_df_diagnosis, self.y_diagnosis)
        # self.adas_model.fit(self.train_df_adas, self.y_adas)
        # self.ventricles_model.fit(self.train_df_ventricles, self.y_ventricles)

    def predict(self, test_df):
        logger.info("Predicting")
        # test_df = self.preprocess(test_series.to_frame().T)

        # select last row per RID
        test_df = test_df.sort_values(by=['EXAMDATE'])
        test_df = test_df.groupby('RID').tail(1)
        exam_dates = test_df['EXAMDATE']
        test_df = self.preprocess(test_df)
        rids = test_df['RID']
        test_df = test_df.drop(['RID'], axis=1)
        test_df = test_df.fillna(0)

        diag_probas = self.diagnosis_model.predict_proba(test_df)
        adas_prediction = self.adas_model.predict(test_df)
        ventricles_prediction = self.adas_model.predict(test_df)

        if self.confidence_intervals:
            logger.info("Bootstrap adas")
            adas_ci = bootstrap(
                self.adas_model,
                self.train_df_adas,
                self.y_adas,
                test_df
            )

            logger.info("Bootstrap ventricles")
            ventricles_ci = bootstrap(
                self.ventricles_model,
                self.train_df_ventricles,
                self.y_ventricles,
                test_df
            )
        else:
            adas_ci = ventricles_ci = 0


        def add_months_to_str_date(strdate, months=1):
            return (datetime.strptime(strdate, '%Y-%m-%d') + relativedelta(months=months)).strftime('%Y-%m-%d')

        df = pd.DataFrame.from_dict({
            'RID': rids,
            'month': 1,
            'Forecast Date': list(map(lambda x: add_months_to_str_date(x, 1), exam_dates.tolist())),
            'CN relative probability': diag_probas.T[0],
            'MCI relative probability': diag_probas.T[1],
            'AD relative probability': diag_probas.T[2],

            'ADAS13': adas_prediction,
            'ADAS13 50% CI lower': adas_prediction - adas_ci,
            'ADAS13 50% CI upper': adas_prediction + adas_ci,

            'Ventricles_ICV': ventricles_prediction,
            'Ventricles_ICV 50% CI lower': ventricles_prediction - ventricles_ci,
            'Ventricles_ICV 50% CI upper': ventricles_prediction + ventricles_ci,
        })

        # copy each row for each month
        new_df = df.copy()
        for i in range(2, 12 * 10):
            df_copy = df.copy()
            df_copy['month'] = i
            df_copy['Forecast Date'] = df_copy['Forecast Date'].map(lambda x: add_months_to_str_date(x, i-1))
            new_df = new_df.append(df_copy)

        return new_df
