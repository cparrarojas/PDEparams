import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error, median_absolute_error
from scipy.integrate import odeint
from scipy.optimize import differential_evolution, minimize
import seaborn as sns

import matplotlib.pyplot as plt
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True

sns.set_palette(["#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
                 "#D55E00", "#CC79A7"])
ncolours = len(plt.rcParams['axes.prop_cycle'])
colours = [list(plt.rcParams['axes.prop_cycle'])[i]['color'] for i in range(ncolours)]

from tqdm.auto import tqdm

class PDEmodel:
    def __init__(self, data, model, initfunc, bounds, param_names=None, nvars=1,
                 ndims=1, nreplicates=1, obsidx=None, outfunc=None):

        '''Initialises the PDEmodel object.

        Parameters
        ----------
        data: DataFrame of data points with each entry in the form:

                                [timepoint, coordinates, fuction values]

        model: the PDE model to fit to the data. Should accept the parameters to
               estimate as its last inputs and return the time derivatives of the
               functions.

        initfunc: array of functions defining the initial conditions for the model.

        bounds: the contraints for the parameter values to use in the estimation,
                as a tuple of tuples or list of tuples.

        param_names (optional, default: None): parameter names to be used in tables
                                               and plots. If None, names appear as
                                               "parameter 1", "parameter 2", etc.

        nvars (optional, default: 1): the number of variables in the system.

        ndims (optional, default: 1): the number of spatial dimensions in the system.

        nreplicates (optional, default: 1): the number of measurements per time-coodinate
                                            pair in the data.

        obsidx (optional, default: None): the indices (starting from zero) of the measured
                                          variables. If None, all outputs are used.

        outfunc (optional, default: None): function to be applied to the output.
                                           If None, raw outputs are used.

        Returns
        -------
        The constructed object.

        '''

        self.model = model
        self.initfunc = initfunc
        self.data = data
        self.bounds = bounds
        self.nvars = nvars
        self.spacedims = ndims
        self.nreplicates = nreplicates
        self.obsidx = obsidx
        self.outfunc = outfunc

        self.nparams = len(self.bounds)

        if param_names is not None:
            self.param_names = param_names
        else:
            self.param_names = ['parameter ' + str(i+1) for i in range(self.nparams)]

        datacols = data.columns.values
        alloutputs = data[datacols[1+ndims:]].values
        allcoordinates = data[datacols[1:1+ndims]].values

        self.timedata = np.sort(np.unique(data[datacols[0]]))
        dt = self.timedata[1] - self.timedata[0]

        self.time = np.concatenate((np.arange(0,self.timedata[0],dt), self.timedata))

        self.timeidxs = np.array([np.argwhere(np.isclose(t, self.time))[0][0] for t in self.timedata])


        if self.spacedims==1:
            self.space = np.sort(np.unique(allcoordinates))
        elif self.spacedims>1:
            shapes = np.empty(self.spacedims).astype(int)
            self.spacerange = []
            grid = []
            for i in range(self.spacedims):
                sortedspace = np.sort(np.unique(allcoordinates[:,i]))
                self.spacerange.append([np.min(sortedspace), np.max(sortedspace)])
                grid.append(sortedspace)
                shapes[i] = sortedspace.shape[0]

            shapes = tuple(np.append(shapes, self.spacedims))

            self.spacerange = np.array(self.spacerange)
            self.space = np.array(np.meshgrid(*(v for v in grid))).T.reshape(shapes)
            self.shapes = shapes

        if self.spacedims == 0:
            self.initial_condition = np.array([self.initfunc[i]() for i in range(self.nvars)])

        elif self.spacedims == 1:
            self.initial_condition = np.array([np.vectorize(self.initfunc[i])(self.space) for i in range(self.nvars)])

        else:
            self.initial_condition = np.array([np.apply_along_axis(self.initfunc[i], -1, self.space) for i in range(self.nvars)])

        if self.nvars == 1:
             self.initial_condition = self.initial_condition[0]

        self.functiondata = alloutputs

        return

    def costfn(self, params, initial_condition, functiondata, bootstrap=False):
        '''Integrates the model and computes the cost function

        Parameters
        ----------
        params: parameter values.

        Returns
        -------
        error: float, the value of the chosen error (see 'fit()') for the given set of parameters.

        '''
        if self.spacedims == 0:
            if self.nparams == 1:
                ft = odeint(self.model, initial_condition, self.time, args=(params[0],))
            else:
                ft = odeint(self.model, initial_condition, self.time, args=tuple(params))

            ft = ft[self.timeidxs]

            if not bootstrap:
                ft = np.repeat(ft, self.nreplicates, axis=0)

            if self.outfunc is not None:
                ft = np.apply_along_axis(self.outfunc, -1, ft)

            elif self.obsidx is not None:
                ft = ft[:, self.obsidx]

            try:
                error = self.error(ft, functiondata)
            except:
                error = np.inf

            if self.sqrt:
                try:
                    error = np.sqrt(error)
                except:
                    error = np.inf

            return error

        else:
            if self.spacedims > 1 or self.nvars > 1:
                initial_condition = initial_condition.reshape(-1)

            ft = odeint(self.model, initial_condition, self.time, args=(self.space, *params))

            if self.nvars>1:
                ft = ft.reshape(ft.shape[0], self.nvars, -1)

                ft = np.array([np.transpose([ft[:,j,:][i] for j in range(self.nvars)]) for i in range(ft.shape[0])])

            if self.spacedims > 1:
                if self.nvars > 1:
                    ft = ft.reshape(ft.shape[0], *self.shapes[:-1], self.nvars)
                else:
                    ft = ft.reshape(ft.shape[0], *self.shapes[:-1])

            ft = ft[self.timeidxs]

            if self.nvars > 1:
                ft = ft.reshape(-1,self.nvars)
            else:
                ft = ft.reshape(-1)

            if not bootstrap:
                ft = np.repeat(ft, self.nreplicates, axis=0)

            if self.outfunc is not None:
                ft = np.apply_along_axis(self.outfunc, -1, ft)

            elif self.obsidx is not None:
                ft = ft[:, self.obsidx]

            try:
                error = self.error(ft, functiondata)
            except:
                error = np.inf

            if self.sqrt:
                try:
                    error = np.sqrt(error)
                except:
                    error = np.inf

            return error

    def fit(self, error='mse'):
        '''Finds the parameters that minimise the cost function using differential evolution.
        Prints them and assigns them to the Estimator object.

        Parameters
        ----------
        error: the type of error to minimise.
            - 'mse': mean squared error
            - 'rmse': root mean squared error
            - 'msle': mean squared logarithmic error
            - 'rmsle': mean squared logarithmic error
            - 'mae': mean absolute error
            - 'medae': median absolute error

        Returns
        -------
        None

        '''

        if error is 'rmse' or error is 'rmsle':
            self.sqrt = True
        else:
            self.sqrt = False

        if error is 'mse' or error is 'rmse':
            self.error = mean_squared_error
        elif error is 'msle' or error is 'rmsle':
            self.error = mean_squared_log_error
        elif error is 'mae':
            self.error = mean_absolute_error
        elif error is 'medae':
            self.error = median_absolute_error

        optimisation = differential_evolution(self.costfn, bounds=self.bounds, args=(self.initial_condition, self.functiondata))

        params = optimisation.x

        best_params = {self.param_names[i]: [params[i]] for i in range(self.nparams)}

        self.best_params = pd.DataFrame(best_params)

        self.best_error = optimisation.fun

        print(self.best_params)
        return

    def likelihood_profiles(self, param_values=None, npoints=100):
        '''Computes the likelihood profile of each parameter.

        Parameters
        ----------
        param_values (optional, default: None): a list of arrays of values for each parameter.
        npoints (optional, default: 100): the number of values in which to divide the parameter
        domain given by the bounds, if param_values is not specified.

        Returns
        -------
        None

        '''

        if not hasattr(self, 'error'):
            self.error = mean_squared_error
            self.sqrt = False

        summary = pd.DataFrame({'parameter': [], 'value': [], 'error': []})
        for i in tqdm(range(self.nparams), desc='parameters'):
            xmin, xmax = self.bounds[i]
            pname = self.param_names[i]

            if param_values is None:
                pvalues = np.linspace(xmin, xmax, npoints)
            else:
                pvalues = param_values[i]

            new_bounds = [bound for bound in self.bounds]

            errors = []

            for pvalue in tqdm(pvalues, desc='values within parameters'):

                new_bounds[i] = (pvalue, pvalue)

                optimisation = differential_evolution(self.costfn, bounds=tuple(new_bounds), args=(self.initial_condition, self.functiondata))

                errors.append(optimisation.fun)

            summary = pd.concat([summary, pd.DataFrame({'parameter': pname, 'value': pvalues, 'error': np.array(errors)})], ignore_index=True)

        self.result_profiles = summary

        return

    def plot_profiles(self):
        '''Plots the likelihood profiles.

        Parameters
        ----------
        None

        Returns
        -------
        None

        '''

        for i, pname in enumerate(self.param_names):
            data = self.result_profiles[self.result_profiles.parameter == pname]

            plt.plot(data.value.values, data.error.values, c=colours[5])
            if np.max(data.error.values) > 250*np.min(data.error.values):
                plt.ylim(-10.*np.min(data.error.values), 250*np.min(data.error.values))
            else:
                plt.ylim(bottom=-3.*np.min(data.error.values))

            if hasattr(self, 'best_params'):
                plt.scatter([self.best_params[pname][0]], [self.best_error], c=colours[1])

            plt.tight_layout()
            plt.xlabel(pname)
            plt.ylabel('error')
            plt.show()

        return

    def bootstrap(self, nruns=100):
        '''Perform bootstrapping by randomly selecting one replicate per time-coodinate pair
        and finding the best fit parameters in multiple rounds.

        Parameters
        ----------
        nruns (optional, default: 100): the number of times to perform the estimation.

        Returns
        -------
        None

        '''
        if not hasattr(self, 'error'):
            self.error = mean_squared_error
            self.sqrt = False

        summary = {self.param_names[i]: [] for i in range(self.nparams)}

        for run in tqdm(range(nruns), desc='runs'):

            idxs = np.arange(self.data.shape[0])[::self.nreplicates]+np.random.randint(self.nreplicates, size=self.data.shape[0]//self.nreplicates)
            data = self.data.iloc[idxs]

            functiondata = self.functiondata[idxs]

            initial_data = data[data[data.columns[0]] == self.timedata[0]]

            optimisation = differential_evolution(self.costfn, bounds=self.bounds, args=(self.initial_condition, functiondata, True))

            params = optimisation.x

            for i in range(self.nparams):
                summary[self.param_names[i]].append(params[i])

        summary = pd.DataFrame(summary)

        self.bootstrap_raw = summary
        self.bootstrap_summary = summary.describe()

        print(self.bootstrap_summary)
        return

    def plot_bootstrap(self):
        '''Plots the bootstrapping results.

        Parameters
        ----------
        None

        Returns
        -------
        None

        '''

        if self.nparams > 1:
            if hasattr(self, 'best_params'):
                data = self.bootstrap_raw.copy()
                data = pd.concat([data, self.best_params], ignore_index=True)
                data['best'] = 0
                data.best.iloc[-1] = 1

                g = sns.pairplot(data, vars=data.columns[:-1], hue='best', palette={0: colours[5], 1: colours[1]}, diag_kind='kde', diag_kws=dict(shade=True))
                g._legend.remove()
            else:
                g = sns.pairplot(self.bootstrap_raw)

        else:
            g = sns.distplot(self.bootstrap_raw, hist=False, kde_kws=dict(shade=True), color=colours[5])
            plt.xlabel(self.param_names[0])
            plt.axvline(x=self.best_params.values[0,0], color=colours[1], linestyle='--', linewidth=1.5)

        plt.tight_layout()
        plt.show()
        return
