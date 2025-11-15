import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import least_squares, minimize
from typing import Dict, Any, List, Optional, Union, Tuple

def compute_numeric_derivative(pricing_fun, base_params, param_names, h=None):
    """
    Numerically compute derivative(s) of pricing_fun at base_params.

    - pricing_fun: callable(**kwargs) -> scalar price
    - base_params: dict of parameters to pass to pricing_fun
    - param_names: str or (str, str)
        * If str -> first derivative w.r.t that parameter (central difference)
        * If (p, p) -> second derivative w.r.t p (second central difference)
        * If (p, q) with p != q -> mixed second partial d2/dp dq (central)
    - h: optional bump size (scalar or dict mapping param->h). If None, automatic h used.

    Returns numeric derivative (float).
    """
    import numpy as np

    def safe_eval(params):
        val = pricing_fun(**params)
        a = np.asarray(val)
        if a.size != 1:
            raise ValueError("pricing_fun must return a scalar-like value")
        return float(a.item())

    eps = np.finfo(float).eps
    # normalize param_names to tuple
    if isinstance(param_names, str):
        params = (param_names,)
    else:
        params = tuple(param_names)

    # helper to compute adaptive h for a single param
    def get_h_for(name, x):
        if h is None:
            return (eps ** (1/3)) * (abs(x) + 1.0)
        if isinstance(h, dict):
            return float(h.get(name, (eps ** (1/3)) * (abs(x) + 1.0)))
        return float(h)

    # single derivative (first)
    if len(params) == 1:
        p = params[0]
        x0 = float(base_params[p])
        hh = get_h_for(p, x0)
        p_plus = dict(base_params); p_plus[p] = x0 + hh
        p_minus = dict(base_params); p_minus[p] = x0 - hh
        f_plus = safe_eval(p_plus)
        f_minus = safe_eval(p_minus)
        return (f_plus - f_minus) / (2.0 * hh)

    # two params -> second derivative or mixed partial
    if len(params) == 2:
        p, q = params
        x_p = float(base_params[p])
        x_q = float(base_params[q])
        h_p = get_h_for(p, x_p)
        h_q = get_h_for(q, x_q)

        if p == q:
            # second derivative w.r.t same parameter
            p_plus = dict(base_params); p_plus[p] = x_p + h_p
            p_minus = dict(base_params); p_minus[p] = x_p - h_p
            f_plus = safe_eval(p_plus)
            f0 = safe_eval(base_params)
            f_minus = safe_eval(p_minus)
            return (f_plus - 2.0 * f0 + f_minus) / (h_p ** 2)
        else:
            # mixed partial: central 4-point formula
            pp = dict(base_params); pp[p] = x_p + h_p; pp[q] = x_q + h_q
            pm = dict(base_params); pm[p] = x_p + h_p; pm[q] = x_q - h_q
            mp = dict(base_params); mp[p] = x_p - h_p; mp[q] = x_q + h_q
            mm = dict(base_params); mm[p] = x_p - h_p; mm[q] = x_q - h_q
            f_pp = safe_eval(pp)
            f_pm = safe_eval(pm)
            f_mp = safe_eval(mp)
            f_mm = safe_eval(mm)
            return (f_pp - f_pm - f_mp + f_mm) / (4.0 * h_p * h_q)

    raise ValueError("param_names must be a string or a tuple/list of two strings")

class BSMModel:
  
    @staticmethod
    def compute_option_with_forward(
        F: float, K: float, T: float, r: float, sigma: float,
        option_type: str, compute_greeks: bool = False
    ) -> Union[float, Dict[str, Any]]:
        """
        Computes the Black-76 price and (optionally) Greeks for a European option on forwards.

        Args:
            F: Forward price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            compute_greeks: If True, returns price and Greeks

        Returns:
            Option price or dict with price and Greeks including vanna and volga
        """
        if T == 0:
            price = max(F - K, 0) if option_type == "call" else max(K - F, 0)
            if not compute_greeks:
                return price
            return {
                "price": price, "delta": np.nan, "gamma": np.nan,
                "vega": np.nan, "theta": np.nan, "vanna": np.nan, "volga": np.nan
            }

        d1 = (np.log(F / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        discount = np.exp(-r * T)

        if option_type == "call":
            price = discount * (F * stats.norm.cdf(d1) - K * stats.norm.cdf(d2))
        elif option_type == "put":
            price = discount * (K * stats.norm.cdf(-d2) - F * stats.norm.cdf(-d1))
        else:
            raise ValueError("Invalid option type. Choose 'call' or 'put'.")

        if not compute_greeks:
            return price

        # Greeks calculations
        delta = discount * stats.norm.cdf(d1) if option_type == "call" else discount * (stats.norm.cdf(d1) - 1)
        gamma = discount * stats.norm.pdf(d1) / (F * sigma * np.sqrt(T))
        vega = discount * F * stats.norm.pdf(d1) * np.sqrt(T) / 100
        theta = (
            -F * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) * discount 
            + r * price
        ) / 252
        
        # Additional Greeks
        vanna = -discount * ((d2 * stats.norm.pdf(d1)) / sigma)  # Vanna computation
        volga = vega * 100 * (d1 * d2 / sigma)  # Volga computation

        return {
            "price": price,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "vanna": vanna,
            "volga": volga
        }

    @staticmethod
    def compute_forward(
        S: float, T: float, r: float, g: float, q: float
        ) -> float:
        """
        Computes the forward price.

        Args:
            S: Spot price
            T: Time to maturity (in years)
            r: Risk-free rate
            g: Growth spread
            q: Dividend yield

        Returns:
            Forward price
        """
        return S * np.exp((r + g - q) * T)

    @staticmethod
    def compute_option_with_spot(
        S: float, K: float, T: float, r: float, g: float, q: float,
        sigma: float, option_type: str, compute_greeks: bool = False
    ) -> Union[float, Dict[str, Any]]:
        """
        Computes the Black-Scholes price and (optionally) Greeks for a European option.

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free rate
            g: Growth spread
            q: Dividend yield
            sigma: Volatility
            option_type: 'call' or 'put'
            compute_greeks: If True, returns price and Greeks

        Returns:
            Option price or dict with price and Greeks
        """
        F = BSMModel.compute_forward(S, T, r, g, q)
        result = BSMModel.compute_option_with_forward(F, K, T, r, sigma, option_type, compute_greeks)
        return result
    
    @staticmethod
    def solve_sigma(
        F: float, K: float, T: float, r: float, market_price: float,
        option_type: str, sigma_init: float = 0.2, tol: float = 1e-5, max_iter: int = 1000
    ) -> float:

        """
        Solves for implied volatility using Newton-Raphson method.

        Args:
            F, K, T, r, market_price, option_type: Option parameters
            sigma_init: Initial guess for volatility
            tol: Tolerance for convergence
            max_iter: Maximum iterations

        Returns:
            Implied volatility or np.nan if not converged
        """
        sigma = sigma_init
        for _ in range(max_iter):
            result = BSMModel.compute_option_with_forward(F, K, T, r, sigma, option_type, compute_greeks=True)
            price = result['price']
            vega = result['vega'] * 100  # Undo /100 in vega
            if vega == 0:
                break
            sigma -= (price - market_price) / vega
            if abs(price - market_price) < tol:
                return sigma
        return np.nan

    @staticmethod
    def compute_montecarlo(
        S: float, T: float, r: float, g: float, q: float,
        sigma: float, n_steps: int, n_paths: int, seed: bool = True, seed_value: Optional[int] = 44
    ) -> pd.DataFrame:
        """
        Simulates asset price paths using Geometric Brownian Motion.

        Args:
            S: Spot price
            T: Time to maturity (in years)
            r: Risk-free rate
            g: Growth spread
            q: Dividend yield
            sigma: Volatility
            n_steps: Number of time steps
            n_paths: Number of simulation paths
            seed: If True, sets random seed for reproducibility

        Returns:
            DataFrame of simulated paths (rows: time steps, columns: paths)
        """
        dt = T / n_steps
        if seed:
            np.random.seed(seed_value)
        dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
        
        S_ts = np.zeros((n_paths, n_steps + 1))
        S_ts[:, 0] = S
        
        for i in range(1, n_steps + 1):
            S_ts[:, i] = S_ts[:, i - 1] * np.exp((r + g - q - 0.5 * sigma ** 2) * dt + sigma * dW[:, i - 1])
        
        return pd.DataFrame(S_ts).transpose()

    @staticmethod
    def compute_pnl_attribution(
        F_start: float, F_end: float,
        K: float, T_start: float, T_end: float,
        r: float, sigma_start: float, sigma_end: float,
        option_type: str
    ) -> Dict[str, float]:
        """
        Computes P&L attribution for an option position using Greeks.
        
        Args:
            F_start: Initial forward price
            F_end: Final forward price
            K: Strike price
            T_start: Initial time to maturity
            T_end: Final time to maturity
            r: Risk-free rate
            sigma_start: Initial volatility
            sigma_end: Final volatility
            option_type: 'call' or 'put'
        
        Returns:
            Dictionary containing P&L components and total P&L
        """
        # Calculate time difference
        dT = T_end - T_start
        
        # Calculate price changes
        dF = F_end - F_start
        dsigma = sigma_end - sigma_start
        
        # Get initial Greeks and price
        start_result = BSMModel.compute_option_with_forward(
            F_start, K, T_start, r, sigma_start, option_type, compute_greeks=True
        )
        
        # Get final price
        end_result = BSMModel.compute_option_with_forward(
            F_end, K, T_end, r, sigma_end, option_type, compute_greeks=False
        )
        
        # Extract Greeks
        delta = start_result['delta']
        gamma = start_result['gamma']
        vega = start_result['vega']
        theta = start_result['theta']
        vanna = start_result['vanna']
        volga = start_result['volga']
        
        # Calculate P&L components
        delta_pnl = delta * dF
        gamma_pnl = 0.5 * gamma * dF * dF
        theta_pnl = (theta * -250) * dT
        vega_pnl = (vega * 100) * dsigma
        vanna_pnl = vanna * dF * dsigma
        volga_pnl = 0.5 * volga * dsigma * dsigma
        
        # Calculate total P&L
        total_pnl = end_result - start_result['price']
        unexplained_pnl = total_pnl - (delta_pnl + gamma_pnl + theta_pnl + 
                                    vega_pnl + vanna_pnl + volga_pnl)
        
        return {
            'total_pnl': total_pnl,
            'delta_pnl': delta_pnl,
            'gamma_pnl': gamma_pnl,
            'theta_pnl': theta_pnl,
            'vega_pnl': vega_pnl,
            'vanna_pnl': vanna_pnl,
            'volga_pnl': volga_pnl,
            'unexplained_pnl': unexplained_pnl
        }

class MultiAssetBSMModel:

    @staticmethod
    def compute_montecarlo(
        S: List[float], T: float, r: float, g: List[float], q: List[float],
        sigma: List[float], rho: Union[List[List[float]], np.ndarray],
        n_steps: int, n_paths: int, seed: bool = True, seed_value: Optional[int] = 44
    ) -> Dict[str, pd.DataFrame]:
        """
        Simulates correlated asset price paths using Geometric Brownian Motion.

        Args:
            S: List of spot prices per asset
            T: Time to maturity (in years)
            r: Risk-free rate
            g: List of growth spreads per asset
            q: List of dividend yields per asset
            sigma: List of volatilities per asset
            rho: Correlation matrix between assets
            n_steps: Number of time steps
            n_paths: Number of simulation paths
            seed: If True, sets random seed for reproducibility

        Returns:
            Dictionary keyed by asset index containing DataFrame of simulated paths
            (rows: time steps, columns: paths)
        """
        S_array = np.asarray(S, dtype=float)
        g_array = np.asarray(g, dtype=float)
        q_array = np.asarray(q, dtype=float)
        sigma_array = np.asarray(sigma, dtype=float)

        n_assets = S_array.size
        if not (g_array.size == q_array.size == sigma_array.size == n_assets):
            raise ValueError("S, g, q, and sigma must have the same length")

        rho_matrix = np.asarray(rho, dtype=float)
        if rho_matrix.shape != (n_assets, n_assets):
            raise ValueError("rho must be a square matrix matching the number of assets")
        if not np.allclose(rho_matrix, rho_matrix.transpose()):
            raise ValueError("rho must be symmetric")

        dt = T / n_steps
        if seed:
            np.random.seed(seed_value)

        try:
            chol = np.linalg.cholesky(rho_matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError("rho must be positive semi-definite") from exc

        standard_normals = np.random.normal(0, 1, (n_steps, n_paths, n_assets))
        paths = np.zeros((n_assets, n_paths, n_steps + 1))
        paths[:, :, 0] = S_array[:, None]

        drift = (r + g_array - q_array - 0.5 * sigma_array ** 2) * dt
        diffusion_scale = sigma_array * np.sqrt(dt)

        for step in range(1, n_steps + 1):
            # Imposes the target correlation structure across assets for this time step
            correlated_shocks = standard_normals[step - 1] @ chol.transpose()
            increments = drift + diffusion_scale * correlated_shocks
            paths[:, :, step] = paths[:, :, step - 1] * np.exp(increments.transpose())

        results: Dict[str, pd.DataFrame] = {}
        for asset_idx in range(n_assets):
            asset_paths = pd.DataFrame(paths[asset_idx].transpose())
            results[f"asset_{asset_idx}"] = asset_paths

        return results

class SABRModel:

    @staticmethod
    def compute_sigma(
        F: float, K: float, T: float, alpha: float, beta: float, rho: float, nu: float
    ) -> float:
        """
        Computes SABR implied volatility using Hagan's formula.

        Args:
            F: Forward price
            K: Strike price
            T: Time to maturity
            alpha, beta, rho, nu: SABR parameters

        Returns:
            Implied volatility
        """
        if F == K:
            factor1 = alpha / (F ** (1 - beta))
            factor2 = 1 + (
                ((1 - beta) ** 2 * alpha ** 2) / (24 * (F ** (2 - 2 * beta)))
                + 0.25 * rho * beta * nu * alpha / (F ** (1 - beta))
                + (2 - 3 * rho ** 2) * nu ** 2 / 24
            ) * T
            return factor1 * factor2

        FK_beta = (F * K) ** ((1 - beta) / 2)
        log_FK = np.log(F / K)
        z = (nu / alpha) * FK_beta * log_FK
        x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
        factor1 = alpha / (FK_beta * (1 + (1 - beta) ** 2 * log_FK ** 2 / 24 + (1 - beta) ** 4 * log_FK ** 4 / 1920))
        factor2 = z / x_z
        factor3 = 1 + (
            ((1 - beta) ** 2 * alpha ** 2) / (24 * (F ** (2 - 2 * beta)))
            + 0.25 * rho * beta * nu * alpha / (F ** (1 - beta))
            + (2 - 3 * rho ** 2) * nu ** 2 / 24
        ) * T
        return factor1 * factor2 * factor3

    @staticmethod
    def compute_option(
        F: float, K: float, T: float, alpha: float, beta: float, rho: float, nu: float,
        r: float, option_type: str, slide_list: Optional[List[float]] = None,
        slide_type: str = 'spot_vol', slide_compute: str = 'delta_hedged_pnl',
        compute_bs_greeks: bool = True, compute_model_greek: bool = False
    ) -> Dict[str, Any]:
        """
        Computes SABR implied volatility and Black-Scholes price/Greeks.

        Args:
            F, K, T, alpha, beta, rho, nu, r: SABR and market parameters
            option_type: 'call' or 'put'
            slide_list: List of spot bumps (optional)
            slide_type: 'spot_vol' or 'spot_only'
            slide_compute: PnL calculation type
            compute_bs_greeks: if True, return BS-76 Greeks
            compute_model_greek: if True, return SABR model Greeks via finite differences

        Returns:
            Dictionary with IV, price, Greeks, and slide results
        """
        def _as_dict(res: Union[float, Dict[str, Any]]) -> Dict[str, Any]:
            return res if isinstance(res, dict) else {"price": res}

        slide_list = slide_list or []
        iv = SABRModel.compute_sigma(F, K, T, alpha, beta, rho, nu)
        base_result = _as_dict(
            BSMModel.compute_option_with_forward(F, K, T, r, iv, option_type, compute_bs_greeks)
        )

        if compute_model_greek:
            base_params = {
                "F": F, "K": K, "T": T, "alpha": alpha,
                "beta": beta, "rho": rho, "nu": nu, "r": r
            }

            def price_scalar(**params):
                local_iv = SABRModel.compute_sigma(
                    params["F"], params["K"], params["T"],
                    params["alpha"], params["beta"], params["rho"], params["nu"]
                )
                return BSMModel.compute_option_with_forward(
                    params["F"], params["K"], params["T"], params["r"],
                    local_iv, option_type, False
                )

            base_result.update({
                "sabr_delta": compute_numeric_derivative(price_scalar, base_params, "F"),
                "sabr_gamma": compute_numeric_derivative(price_scalar, base_params, ("F", "F")),
                "sabr_vega": compute_numeric_derivative(price_scalar, base_params, "alpha"),
                "sabr_vanna": compute_numeric_derivative(price_scalar, base_params, "rho"),
                "sabr_volga": compute_numeric_derivative(price_scalar, base_params, "nu"),
                "sabr_theta": compute_numeric_derivative(price_scalar, base_params, "T"),
            })

        for slide in slide_list:
            if slide_type == 'spot_vol':
                F_bumped = F * (1 + slide)
                dsigma = (nu / alpha) * rho * slide
                alpha_bumped = alpha * (1 + dsigma)
                iv_bumped = SABRModel.compute_sigma(F_bumped, K, T, alpha_bumped, beta, rho, nu)
                bumped_result = _as_dict(
                    BSMModel.compute_option_with_forward(
                        F_bumped, K, T, r, iv_bumped, option_type, compute_bs_greeks
                    )
                )
            elif slide_type == 'spot_only':
                F_bumped = F * (1 + slide)
                iv_bumped = SABRModel.compute_sigma(F, K, T, alpha, beta, rho, nu)
                bumped_result = _as_dict(
                    BSMModel.compute_option_with_forward(
                        F_bumped, K, T, r, iv_bumped, option_type, compute_bs_greeks
                    )
                )
            else:
                continue
            if slide_compute == 'delta_hedged_pnl':
                option_pnl = bumped_result['price'] - base_result['price']
                delta_value = base_result.get('delta', np.nan)
                delta_hedge_pnl = F * delta_value * slide * -1
                total_pnl = delta_hedge_pnl + option_pnl
                base_result[slide] = total_pnl
            elif slide_compute == 'option_pnl':
                base_result[slide] = bumped_result['price'] - base_result['price']
            elif slide_compute == 'delta_pnl':
                delta_value = base_result.get('delta', np.nan)
                base_result[slide] = F * delta_value * slide * -1
            else:
                base_result[slide] = bumped_result.get(slide_compute, np.nan)

        return {'IV': iv, **base_result}

    @staticmethod
    def solve_delta_strike(
        F: float, T: float, alpha: float, beta: float, rho: float, nu: float,
        r: float, option_type: str, target_delta: float
    ) -> float:
        """
        Finds the strike corresponding to a target delta.

        Args:
            F, T, alpha, beta, rho, nu, r: SABR and market parameters
            option_type: 'call' or 'put'
            target_delta: Desired delta

        Returns:
            Strike value
        """
        def objective(K):
            result = SABRModel.compute_option(F, K[0], T, alpha, beta, rho, nu, r, option_type)
            return (result['delta'] - target_delta) ** 2

        res = minimize(objective, x0=[F], bounds=[(F * 0.01, F * 20.0)], method='L-BFGS-B')
        return res.x[0]

    @staticmethod
    def compute_varswap(
        F: float, T: float, alpha: float, beta: float, rho: float, nu: float,
        r: float, K_min: int, K_max: int
    ) -> float:
        """
        Computes the fair value of a variance swap using SABR model.

        Args:
            F, T, alpha, beta, rho, nu, r: SABR and market parameters
            K_min, K_max: Range of strikes

        Returns:
            Variance swap value
        """
        vs = {}
        for K in range(K_min, K_max, 1):
            option_type = 'call' if K >= F else 'put'
            pv = SABRModel.compute_option(F, K, T, alpha, beta, rho, nu, r, option_type)['price']
            vs[K] = pv
        vs_df = pd.Series(vs).to_frame('pv')
        vs_df.index.name = 'k'
        vs_df = vs_df.reset_index()
        vs_df['dk'] = vs_df['k'].diff().fillna(0)
        k_var = np.sum((vs_df['pv'] / vs_df['k'].pow(2)) * vs_df['dk']) * (2 / T)
        return k_var

    @staticmethod
    def compute_montecarlo(
        F: float, T: float, alpha: float, beta: float, rho: float, nu: float,
        n_steps: int, n_paths: int, seed: bool = True, seed_value: Optional[int] = 44
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simulates SABR paths using Euler-Maruyama.

        Args:
            F, T, alpha, beta, rho, nu: SABR parameters
            n_steps: Number of time steps
            n_paths: Number of simulation paths
            seed: If True, sets random seed for reproducibility

        Returns:
            Tuple of DataFrames: (F_paths, sigma_paths)
        """
        dt = T / n_steps
        if seed:
            np.random.seed(seed_value)
            dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
            np.random.seed(seed_value + 1)
            dZ = rho * dW + np.sqrt(1 - rho ** 2) * np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
        else:
            dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
            dZ = rho * dW + np.sqrt(1 - rho ** 2) * np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))

        F_ts = np.zeros((n_paths, n_steps + 1))
        sigma_ts = np.zeros((n_paths, n_steps + 1))
        F_ts[:, 0] = F
        sigma_ts[:, 0] = alpha

        for i in range(1, n_steps + 1):
            F_ts[:, i] = F_ts[:, i - 1] + sigma_ts[:, i - 1] * F_ts[:, i - 1] ** beta * dW[:, i - 1]
            sigma_ts[:, i] = sigma_ts[:, i - 1] * np.exp(nu * dZ[:, i - 1] - 0.5 * nu ** 2 * dt)

        return pd.DataFrame(F_ts).transpose(), pd.DataFrame(sigma_ts).transpose()

    @staticmethod
    def solve_parameters(
        F: float, T: float, strikes: List[float], market_vols: List[float],
        init_guess: List[float] = [0.1, 0.0, 0.3],
        lower_bounds: List[float] = [1e-6, -0.9999, 1e-6],
        upper_bounds: List[float] = [1, 0.9999, 3]
    ) -> Tuple[float, float, float]:
        """
        Calibrates SABR parameters (alpha, rho, nu) to market volatilities.

        Args:
            F, T: Market parameters
            strikes: List of strikes
            market_vols: List of market implied vols
            init_guess: Initial guess for [alpha, rho, nu]
            lower_bounds, upper_bounds: Parameter bounds

        Returns:
            Tuple: (alpha, rho, nu)
        """
        def objective(params, F, strikes, T, market_vols):
            alpha, rho, nu = params
            model_vols = [
                SABRModel.compute_sigma(F, K, T, alpha, beta=1, rho=rho, nu=nu)
                for K in strikes
            ]
            return np.array(market_vols) - np.array(model_vols)

        bounds = (lower_bounds, upper_bounds)
        result = least_squares(objective, x0=init_guess, args=(F, strikes, T, market_vols), bounds=bounds)
        return tuple(result.x)

    @staticmethod
    def solve_alpha(
        F: float, T: float, rho: float, nu: float, r: float,
        K_min: int, K_max: int, K_var: float,
        init_guess: List[float] = [0.1], lower_bounds: List[float] = [1e-6], upper_bounds: List[float] = [1]
    ) -> float:
        """
        Solves for SABR alpha parameter to match a target variance swap value.

        Args:
            F, T, rho, nu, r: SABR and market parameters
            K_min, K_max: Range of strikes
            K_var: Target variance swap value
            init_guess, lower_bounds, upper_bounds: Optimization parameters

        Returns:
            Calibrated alpha
        """
        def objective(alpha, F, T, rho, nu, r, K_min, K_max, K_var):
            return K_var - SABRModel.compute_varswap(F, T, alpha, beta=1, rho=rho, nu=nu, r=r, K_min=K_min, K_max=K_max)

        bounds = (lower_bounds, upper_bounds)
        result = least_squares(objective, x0=init_guess, args=(F, T, rho, nu, r, K_min, K_max, K_var), bounds=bounds)
        return result.x[0]
      
class HestonHullWhiteModel:

    @staticmethod
    def compute_montecarlo(
        S0: float, v0: float, r0: float, b0: float,
        kappa_v: float, theta_v: float, sigma_v: float,
        kappa_r: float, theta_r: float, sigma_r: float,
        rho_sv: float, rho_sr: float, rho_vr: float,
        T: float, N: int, M: int,
        seed: bool = True, seed_value: Optional[int] = 44
        ) -> Dict[int, Dict[str, np.ndarray]]:
    
        """
        Simulate Monte Carlo paths under the Heston-Hull-White model with risk control,
        allowing an override of the first Brownian increment dW1 at t=1 while preserving
        the correlation structure with dW2 and dW3.

        If override_dW1 is provided, it will replace the random dW1 at t=1 for all paths.
        """
        dt = T / M

        # Build correlation matrix and its Cholesky factor
        corr = np.array([
            [1.0, rho_sv, rho_sr],
            [rho_sv, 1.0, rho_vr],
            [rho_sr, rho_vr, 1.0]
        ])
        
        L = np.linalg.cholesky(corr)

        # Allocate arrays
        S = np.zeros((M + 1, N))
        v = np.zeros((M + 1, N))
        r = np.zeros((M + 1, N))
        S[0, :] = S0
        v[0, :] = v0
        r[0, :] = r0

        if seed:
            rng = np.random.default_rng(seed_value)
        else:
            rng = np.random.default_rng()

        sqrt_dt = np.sqrt(dt)

        for t in range(1, M + 1):

            # Generate independent standard normals
            Z = rng.standard_normal((3, N))

            # Compute correlated increments
            dW = L @ Z * sqrt_dt
            dW1, dW2, dW3 = dW[0], dW[1], dW[2]

            # Full-truncation Euler for variance
            v_prev = np.maximum(v[t-1, :], 0.0)
            v[t, :] = (
                v[t-1, :]
                + kappa_v * (theta_v - v_prev) * dt
                + sigma_v * np.sqrt(v_prev) * dW2
            )
            v[t, :] = np.maximum(v[t, :], 0.0)

            # Hull-White short rate
            r[t, :] = (
                r[t-1, :]
                + kappa_r * (theta_r - r[t-1, :]) * dt
                + sigma_r * dW3
            )

            # Asset price log-Euler
            S[t, :] = (
                S[t-1, :]
                * np.exp((r[t-1, :] + b0 - 0.5 * v_prev) * dt + np.sqrt(v_prev) * dW1)
            )
        
        return S, v, r
