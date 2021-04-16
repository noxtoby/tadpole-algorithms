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

import os

from cmdstanpy import cmdstan_path, CmdStanModel

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
    """Differential Equation Model method, Neil Oxtoby (Team Billabong) - neil@neiloxtoby.com

    The `train_df*` attributes contain training data optimized for each variable.

    The `y_train_df*` attributes contain the labels to be used for training by each model,
    thus corresponding to the matching `train_df` DataFrame.

    Attributes:
        diagnosis_model (Pipeline): Model for predicting 'diagnosis' variable
        adas_model (Pipeline): Model for predicting 'ADAS13' variable
        ventricles_model (Pipeline): Model for predicting 'ventricles' variable

        y_diagnosis (pandas.DataFrame): 'Diagnosis' labels
        train_df_diagnosis (pandas.DataFrame): Training data used for 'diagnosis' model.

    Original TADPOLE Challenge submission generated with Neil's MATLAB code:
        TADPOLE_Oxtoby_DEM.m and dependencies
        ====== The DEM pipeline for a single biomarker ======
        0.  Check for existing results before proceeding
        1.  Refine training data: APOE4+ clinical progressors or stable AD
        2.  Post-select on coefficient of variation (std/mean < 0.25 for each individual)
        3.  Calculate differential data
        4.  (Optional) Remove normal and non-progressing differential data
        5.  Fit DEM: Gaussian Process Regression
        6.  Integrate fit probabilistically: samples from GPR posterior
        7.  Anchor trajectory to median biomarker value for stable AD (first visit): anchorTrajectory(x_anchor)
        9.  (Optional) Plot: DEM.plotDEM()
        10. Save results
        11. Analyse results (possibly write DEM.analyseResults()):
            a) Fitting: MCMC traces, etc.
               For example: fit.plot where fit = pystan.stan(); or my own DEM.plotDEM()
            b) Patient staging (e.g., at baseline): tStage = DEM.stageIndividuals(xValuesToStage,id,DEM)
            c) Prediction of symptom onset for clinical progressors
    
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
        """
        preprocess(train_df):
            - Various cleaning: DX, Age
            - 
        """
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

        dem_markers = [
            'Ventricles_ICV','Hippocampus_ICV','WholeBrain_ICV','Entorhinal_ICV','Fusiform_ICV','MidTemp_ICV',
            'MMSE', 'ADAS13'
        ]
        
        # Divide certain volumes by ICV. Could add more...
        #* Avoid divide-by-zero errors: code by Esther Bron
        icv_bl_median = train_df['ICV_bl'].median()
        train_df.at[train_df['ICV_bl'] == 0, 'ICV_bl'] = icv_bl_median

        for vol in ['Ventricles','Hippocampus','WholeBrain','Entorhinal','Fusiform','MidTemp']:
            train_df[vol+'_ICV'] = train_df[vol].values/train_df['ICV_bl'].values
            #test_df[vol+'_ICV'] = test_df[vol].values/test_df['ICV_bl'].values

        """Sort the DataFrame per patient on age (at time of visit). This allows using observations from
        the next row/visit to be used as a label for the previous row. (See `get_futures` method.)"""
        train_df = train_df.sort_values(by=['RID', 'VISITAGE'])

        #train_df = train_df.drop(['EXAMDATE', 'AGE', 'PTGENDER', 'PTEDUCAT', 'APOE4'], axis=1)
        """Select features based on Billabong_features.csv file"""
        #FIXME: update Billabong_features.csv to include the following:
        # biomarkers = ['Ventricles_ICV','Hippocampus_ICV','WholeBrain_ICV','Entorhinal_ICV','Fusiform_ICV','MidTemp_ICV',
        #     'FDG','ABETA_UPENNBIOMK9_04_19_17','PTAU_UPENNBIOMK9_04_19_17','TAU_UPENNBIOMK9_04_19_17','ADAS13','MMSE','MOCA','RAVLT_immediate']
        features_csv_path = Path(__file__).parent / 'Billabong_features.csv' 
        selected_features = pd.read_csv(features_csv_path)['feature'].values.tolist()
        train_df = train_df.copy()[set(['RID', 'Diagnosis', 'Years_bl'] + selected_features)]

        logger.info('Forcing Numeric Values')
        for f in selected_features:
            if train_df[f].dtype != 'float64':
                train_df[f] = pd.to_numeric(train_df[f], errors='coerce')

        logger.info('Filling missing NaNs with older values')
        train_df = DEM.fill_nans_by_older_values(train_df)

        #* Preprocessing for DEM
        logger.info('Preprocessing for DEM model:')

        logger.info('  ...DEM.dem_gradients(): calculating rate of change for each biomarker')
        df_dem = DEM.dem_gradients(
            df=train_df,
            markers=selected_features,
            t_col='Years_bl',
            dx_col='Diagnosis'
        )

        logger.info('  ...DEM.findStablesAndProgressors(): identifying clinical progressors')
        stable, progressor, reverter, mixed, \
            progression_visit, reversion_visit, \
            stable_u, progressor_u, reverter_u, mixed_u, progression_visit_u, reversion_visit_u = \
            DEM.findStablesAndProgressors(df_dem.Years_bl,df_dem.DXNUM,df_dem.RID)
        df_dem.stables          = stable
        df_dem.progressors      = progressor
        df_dem.progressionVisit = progression_visit

        #* Controls: reference group for calculating correction factors (regression)
        df_dem.normals   = stable==1 & df_dem.APOE4==0 & df_dem.DXNUM==11 & df_dem.CDRSB==0
        #* Patients: group upon which to build the DEM
        df_dem.abnormals = stable==1 & df_dem.APOE4==1 & df_dem.DXNUM==33

        logger.info('  ...DEM.dem_postselect(): biomarker-based postselection to exclude non-progressors, non-abnormal')
        df_dem_, df_postselection = DEM.dem_postselect(df_dem=df_dem,markers=selected_features,dx_col='Diagnosis.bl')

        #* Optionally subsample to reduce the dimensions (fitting can take quite some time)
        subsample = 3
        #- Sort biomarkers to approximate getting even coverage
        df_dem__ = df_dem_.sort_values(by=[d+'-mean' for d in selected_features])
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
                df_dem.at[rowz_dem,m+'-mean'] = np.mean(x)
                df_dem.at[rowz_dem,m+'-grad'] = DEM.dxdt(x,t)
                #* Calculate coefficient of variation
                df_dem.at[rowz_dem,m+'-coef_of_variation'] = np.std(x)/np.mean(x)
                #train_df[['RID',m]].groupby(by='RID').std()/train_df[['RID',m]].groupby(by='RID').mean()
        return df_dem

    @staticmethod
    def findStablesAndProgressors(t,diagnosis_numerical,id):
        "findStablesAndProgressors(t,diagnosis_numerical,id): Loops through to find subjects who progress clinically (and those who don't)."
        # Unique id
        id_u = np.unique(id)
        # Progressor, Stable, Reverter, Mixed
        progressor_u = np.zeros(id_u.shape, dtype=bool)
        stable_u     = np.zeros(id_u.shape, dtype=bool)
        reverter_u   = np.zeros(id_u.shape, dtype=bool)
        mixed_u      = np.zeros(id_u.shape, dtype=bool)

        progression_visit_u    = np.empty(id_u.shape)
        progression_visit_u[:] = np.nan
        reversion_visit_u      = np.empty(id_u.shape)
        reversion_visit_u[:]   = np.nan

        n_visits_u    = np.empty(id_u.shape)
        n_visits_u[:] = np.nan

        progressor = np.zeros(id.shape, dtype=bool)
        stable     = np.zeros(id.shape, dtype=bool)
        reverter   = np.zeros(id.shape, dtype=bool)
        mixed      = np.zeros(id.shape, dtype=bool)

        progression_visit = np.zeros(id.shape, dtype=bool)
        reversion_visit   = np.zeros(id.shape, dtype=bool)

        n_visits = np.empty(id.shape)

        # Loop through id and identify subjects who progress in diagnosis
        for k in range(0,len(id_u)):
            rowz = id==id_u[k]
            tee = t[rowz]
            dee = diagnosis_numerical[rowz]
            rowz_f = np.where(rowz)[0]

            #= Missing data: should be superfluous (you should've handled it already)
            not_missing = np.logical_and( np.isnan(dee)==False , np.isnan(tee)==False )
            dee = dee[not_missing]
            tee = tee[not_missing]
            rowz_f = rowz_f[not_missing]

            #= Number of visits
            n_visits_u[k] = len(dee) #sum( not_missing )

            #= Order diagnosis in time
            ordr = np.argsort(tee)
            dee = dee[ordr]

            #= if longitudinal data exists for this individual
            if len(dee)>1:
                dee_diff = np.diff(dee)
                if all(dee_diff>=0) & any(dee_diff>0):
                    #= if Progressor
                    progressor_u[k] = True
                    #=== Identify progression visits ===
                    pv = np.where(dee_diff>0)[0] # all visits where progression occurs
                    progression_visit_u[k] = pv[0] + 1 # +1 to account for the np.diff()
                    progression_visit[rowz_f[np.int(progression_visit_u[k])]] = True
                elif all(dee_diff==0):
                    #= if Stable
                    stable_u[k] = True
                elif all(dee_diff<0): 
                    #= if Reverter
                    reverter_u[k] = True
                    #=== Identify reversion visits ===
                    rv = np.where(dee_diff<0)[0] # all visits where reversion occurs
                    reversion_visit_u[k] = rv[0] + 1 # +1 to account for the np.diff()
                    reversion_visit[rowz_f[np.int(reversion_visit_u[k])]] = True
                else:
                    #= if mixed diagnosis (both progression and reversion)
                    mixed_u[k] = True
            #=== Propagate individual data back to original shape vectors ===
            n_visits[rowz] = n_visits_u[k]
            progressor[rowz] = progressor_u[k]
            stable[rowz] = stable_u[k]
            reverter[rowz] = reverter_u[k]
            mixed[rowz] = mixed_u[k]

    return stable, progressor, reverter, mixed, progression_visit, reversion_visit, stable_u, progressor_u, reverter_u, mixed_u, progression_visit_u, reversion_visit_u

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

            df_postelection.at[df_postelection['Marker']==m,'Normal-Threshold'] = normal_threshold

        return df_, df_postelection

    @staticmethod
    def fit_dem(df_dem,stan_model):
        """
        dem_fit = fit_dem(df_dem,stan_model)
        """
        x_text = '-mean'
        y_text = '-grad'
        cofv_text = '-coef_of_variation'
        markers = df_dem.columns.tolist()
        markers = [m.replace(x_text,'') for m in markers if 'mean' in m]

        df_dem_fits = pd.DataFrame(data={'Marker':markers,'x_predict':None})

        # #* 1. Linear regression
        # slope, intercept, r_value, p_value, std_err = stats.linregress(x_,dxdt_)
        # DEMfit = {'linreg_slope':slope}
        # DEMfit['linreg_intercept'] = intercept
        # DEMfit['linreg_r_value'] = r_value
        # DEMfit['linreg_p_value'] = p_value
        # DEMfit['linreg_std_err'] = std_err

        for m in markers:
            txt = 'Performing GPR for {0}'.format(m)
            print(txt)
            logger.info(txt)
            cofv_okay = df_dem[m+cofv_text].values < 0.25
            x = df_dem[m+x_text].values[cofv_okay]
            y = df_dem[m+y_text].values[cofv_okay]
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
            #df_dem_fits.at[df_dem_fits['Marker']==m,'x_predict'] = x_predict
            fit = stan_model.sample(data=dem_gpr_dat,
                                    inits=init, #[init,init,init,init],
                                    iter_warmup=500, iter_sampling=1000,
                                    parallel_chains=4)
            df_dem_fits.at[df_dem_fits['Marker']==m,'stan_fit_gpr'] = fit
        return df_dem_fits

    @staticmethod
    def sample_from_gpr_posterior(x,y,xp,alpha,rho,sigma,
            CredibleIntervalLevel=0.95,
            nSamplesFromGPPosterior=500):
        #* GP Posterior
        stds = np.sqrt(2) * special.erfinv(CredibleIntervalLevel)
        #* Covariance matrices from kernels: @kernel_pred, @kernel_err, @kernel_obs
        def kernel_pred(alpha,rho,x_1,x_2):
            kp = alpha**2*np.exp(-rho**2 * (np.tile(x_1,(len(x_2),1)).transpose() - np.tile(x_2,(len(x_1),1)))**2)
            return kp
        def kernel_err(sigma,x_1):
            ke = sigma**2*np.eye(len(x_1))
            return ke
        def kernel_obs(alpha,rho,sigma,x_1):
            ko = kernel_pred(alpha,rho,x_1,x_1) + kernel_err(sigma,x_1)
            return ko
        #* Observations - full kernel
        K = kernel_obs(alpha=alpha,rho=rho,sigma=sigma,x_1=x)
        #* Interpolation - signal only
        K_ss = kernel_pred(alpha=alpha,rho=rho,x_1=xp,x_2=xp)
        #* Covariance (observations & interpolation) - signal only
        K_s = kernel_pred(alpha=alpha,rho=rho,x_1=xp,x_2=x)
        #* GP mean and covariance
        #* Covariance from fit
        y_post_mean = np.matmul(np.matmul(K_s,np.linalg.inv(K)),y)
        y_post_Sigma = (K_ss - np.matmul(np.matmul(K_s,np.linalg.inv(K)),K_s.transpose()))
        y_post_std = np.sqrt(np.diag(y_post_Sigma))
        #* Covariance from data - to calculate residuals
        K_data = K
        K_s_data = kernel_pred(alpha=alpha,rho=rho,x_1=x,x_2=x)
        y_post_mean_data = np.matmul(np.matmul(K_s_data,np.linalg.inv(K_data)),y)
        residuals = y1 - y_post_mean_data
        rmse = np.sqrt(np.mean(residuals**2))

        # Numerical precision
        eps = np.finfo(float).eps

        ## 3. Sample from the posterior (multivariate Gaussian)
        #* Diagonalise the GP posterior covariance matrix
        Vals,Vecs = np.linalg.eig(y_post_Sigma)
        A = np.real(np.matmul(Vecs,np.diag(np.sqrt(Vals))))

        y_posterior_middle = y_post_mean
        y_posterior_upper = y_post_mean + stds*y_post_std
        y_posterior_lower = y_post_mean - stds*y_post_std

        #* Sample
        y_posterior_samples = np.tile(y_post_mean,reps=(nSamplesFromGPPosterior,1)).transpose() 
                            + np.matmul(A,np.random.randn(len(y_post_mean),nSamplesFromGPPosterior))
        if np.abs(np.std(y)-1) < eps:
            y_posterior_samples = y_posterior_samples*np.std(y) + np.mean(y)
    
        return (y_posterior_middle,y_posterior_upper,y_posterior_lower,y_posterior_samples)

    @staticmethod
    def integrate_dem(df_dem_fits):
        """
        WIP:
        
        dem_trajectories = integrate_dem(df_dem_fits)
        
        Integrates GPR DEM into trajectories
        """

        dem_markers = df_dem_fits.columns.tolist()
        for m in dem_markers:
            fit = df_dem_fits.at[df_dem_fits['Marker']==m,'stan_fit_gpr'].values[0]
            x = fit.data['x']
            y = fit.data['y']
            x_predict = fit.data['x_predict']

            rho   = fit.stan_variable('rho')
            alpha = fit.stan_variable('alpha')
            sigma = fit.stan_variable('sigma')
            # samples = fit.extract(permuted=True, inc_warmup=False)
            # rho = np.median(samples['rho'],0)
            # alpha = np.median(samples['alpha'],0)
            # sigma = np.median(samples['sigma'],0)

            #* These ones may need CmdStanModel.generate_quantities()
            f_predict = fit.stan_variable('f_predict')
            y_predict = fit.stan_variable('y_predict')
            #f_predict = np.median(samples['f_predict'],0) # GP posterior mean
            #y_predict = np.median(samples['y_predict'],0) # GP posterior spread

            (y_posterior_middle,y_posterior_upper,y_posterior_lower,y_posterior_samples) = DEM.sample_from_gpr_posterior(
                x,y,x_predict,alpha,rho,sigma,
                CredibleIntervalLevel=0.95,
                nSamplesFromGPPosterior=500
            )

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

    @staticmethod
    def coef_of_var(x,vis,id):
        """
        x_cv = coef_of_var(x,vis,id)
           Calculate coefficient of variation within-subject (longitudinal scalar biomarker data).
           x   = data (e.g., biomarker values)
           vis = visit numbers
           id  = individual ID
        Neil Oxtoby, UCL, Nov 2017
        """
        x_matrix  = convertVectorsToMatrix(x,vis,id)
        id_matrix = convertVectorsToMatrix(id,vis,id)

% [~,~,vis2] = unique(vis,'stable'); % Make sure no zeros

x_cv_vector = nanstd(x_matrix,1)./nanmean(x_matrix,1);
id_cv_vector = nanmean(id_matrix,1);

x_cv = nan(size(x));
% for k=1:length(x_cv)
%   if ~isnan(x(k)) && ~isnan(vis(k)) && ~isnan(id(k))
%     x_cv(k) = x_cv_vector(id(k));
%   end
% end
for ki=1:length(id_cv_vector)
  rowz = id == id_cv_vector(ki);
  cv = x_cv_vector(ki);
  n = ~isnan(x(rowz)) & ~isnan(vis(rowz)) & ~isnan(id(rowz));
  rowz = find(rowz);
  x_cv(rowz(n)) = cv;
end

end



    def train(self, train_df):
        logger.info("Training model(s)")

        df_dem = self.preprocess(train_df)

        #* Include only clinical progressors
        logger.info("Refining training data: APOE4+ clinical progressors and stable AD")
        clinicalProgressors = df_dem.progressors
        apoe4 = df_dem.APOE4.astype(float)
        apoe4 = apoe4 - np.min(apoe4)
        DEMsubset = (df_dem.abnormals==1 | clinicalProgressors==1) & apoe4>0
        df_dem = df_dem.loc[DEMsubset].copy()

        # Prep for Gaussian Process regression
        logger.info("Differential Equation Model fitting: GP regression")
        gpr_stan_code = Path(__file__).parent / 'gp_betancourt.stan'
        #dem_gpr_stan_model = pystan.StanModel(file=gpr_stan_code.as_posix(),model_name='gpr_dem') # pystan doesn't work in jupyter notebooks!
        dem_gpr_stan_model = CmdStanModel(stan_file=gpr_stan_code.as_posix(),model_name='gpr_dem')
        # Check for saved results and fit
        fname_save_stan_fits = "DEM-results-pickle-stan-fits"
        check_flag = check_for_save_file(file_name=fname_save_stan_fits,function=None)
        if check_flag: #"dem_gpr_stan_model_fits" in stan_results:
            logger.info('DEM.train():           Not fitting DEMs: existing results detected.')
            pickle_file = open(fname_save_stan_fits,'rb')
            stan_results = pickle.load(pickle_file)
            dem_gpr_stan_model_fits = stan_results["dem_gpr_stan_model_fits"]
            pickle_file.close()
            #dem_gpr_stan_model_fits = stan_results["dem_gpr_stan_model_fits"]
        else:
            logger.info('DEM.train():           Removing values having high coefficient of variation')
            removeLargeCoeffOfVariance = True
            #* Coefficient of variation < 0.25 (see Bateman 2012, and possibly also Fagan 2014: longitudinal CSF)
            if removeLargeCoeffOfVariance:
                train_df[['RID','Ventricles','Hippocampus']].groupby(by='RID').std()/train_df[['RID','Ventricles','Hippocampus']].groupby(by='RID').mean()
                
                x_CoV = TADPOLE_Oxtoby_CoefOfVar(x,t,id);
              coeffOfVariationPostselect = x_CoV < 0.25;
              DEM_subset = DEM_subset & coeffOfVariationPostselect;
            end
            
            logger.info('DEM.train():           Fitting DEMs using GP regression')
            self.df_dem_fits = self.fit_dem(df_dem,dem_gpr_stan_model)
            #* Save the fits to a new pickle file - the StanModel mmust be unpickled first
            stan_results["dem_gpr_stan_model_fits"] = self.df_dem_fits
            pickle_file = open(fname_save_stan_fits,'wb')
            pickle_output = pickle.dump(stan_results,pickle_file)
            pickle_file.close()

        # self.diagnosis_model.fit(self.train_df_diagnosis, self.y_diagnosis)
        # self.adas_model.fit(self.train_df_adas, self.y_adas)
        # self.ventricles_model.fit(self.train_df_ventricles, self.y_ventricles)

    def predict(self, test_df):
        #* TODO: change to DEM-specific predict code (translate from MATLAB)
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
