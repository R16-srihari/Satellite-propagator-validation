# Integrator Comparison Report

Comparison of **pd853** vs **symplectic** integrator conservation
errors against the analytical two-body reference.

## Orbit Parameters

- Semi-major axis: **6.828137e+06 m**
- Eccentricity: **0.0** (circular)
- Inclination: **51.6 deg**
- Orbital period: **5615.19 s** (93.59 min)
- Specific orbital energy: **-2.918808e+07 J/kg**
- Angular momentum magnitude: **5.216990e+10 m^2/s**
- Epoch (UTC): **2026-05-18 06:30:00**

## Methodology

Both integrators share the same time grid (0-86400 s, 10 s steps).
Errors are computed against the closed-form Keplerian two-body
solution using `compare_analytical()`. This report overlays the
per-time-step errors for each metric.

## Summary Statistics (RMS error)

| Metric | Unit | pd853 RMS | symplectic RMS | Ratio (sym/pd853) |
|--------|------|-----------|----------------|--------------------|
| Position error ||r|| | m | 3.402971e-06 | 2.029933e-06 | 0.5965 |
| Position relative error | 1 | 4.983747e-13 | 2.972895e-13 | 0.5965 |
| Velocity error ||v|| | m/s | 3.805628e-09 | 2.263238e-09 | 0.5947 |
| Velocity relative error | 1 | 4.980908e-13 | 2.962186e-13 | 0.5947 |
| Angular momentum error |h| | m^2/s | 1.950204e-04 | 2.189711e-04 | 1.1228 |
| Angular momentum relative error | 1 | 3.738179e-15 | 4.197269e-15 | 1.1228 |
| Energy error | J/kg | 2.187054e-07 | 2.453135e-07 | 1.1217 |
| Energy relative error | 1 | 7.492970e-15 | 8.404577e-15 | 1.1217 |

## Detailed Statistics

| Integrator | Metric | Max | Mean | RMS |
|------------|--------|-----|------|-----|
| pd853 | Angular momentum error |h| | 3.738403e-04 | 1.717598e-04 | 1.950204e-04 |
| symplectic | Angular momentum error |h| | 4.272461e-04 | 1.907199e-04 | 2.189711e-04 |
| pd853 | Angular momentum relative error | 7.165824e-15 | 3.292317e-15 | 3.738179e-15 |
| symplectic | Angular momentum relative error | 8.189513e-15 | 3.655745e-15 | 4.197269e-15 |
| pd853 | Energy error | 4.209578e-07 | 1.927209e-07 | 2.187054e-07 |
| symplectic | Energy error | 4.917383e-07 | 2.137698e-07 | 2.453135e-07 |
| pd853 | Energy relative error | 1.442225e-14 | 6.602727e-15 | 7.492970e-15 |
| symplectic | Energy relative error | 1.684723e-14 | 7.323874e-15 | 8.404577e-15 |
| pd853 | Position error ||r|| | 6.585053e-06 | 2.624105e-06 | 3.402971e-06 |
| symplectic | Position error ||r|| | 4.517047e-06 | 1.644643e-06 | 2.029933e-06 |
| pd853 | Position relative error | 9.643996e-13 | 3.843076e-13 | 4.983747e-13 |
| symplectic | Position relative error | 6.615344e-13 | 2.408626e-13 | 2.972895e-13 |
| pd853 | Velocity error ||v|| | 7.341438e-09 | 2.934437e-09 | 3.805628e-09 |
| symplectic | Velocity error ||v|| | 4.801078e-09 | 1.835414e-09 | 2.263238e-09 |
| pd853 | Velocity relative error | 9.608671e-13 | 3.840670e-13 | 4.980908e-13 |
| symplectic | Velocity relative error | 6.283781e-13 | 2.402240e-13 | 2.962186e-13 |

## Per-Metric Winner

| Metric | Better Integrator |
|--------|-------------------|
| Position error ||r|| | symplectic |
| Position relative error | symplectic |
| Velocity error ||v|| | symplectic |
| Velocity relative error | symplectic |
| Angular momentum error |h| | pd853 |
| Angular momentum relative error | pd853 |
| Energy error | pd853 |
| Energy relative error | pd853 |

## Generated Plots

- `comparative_angular_momentum_error.png`
- `comparative_energy_error.png`
- `comparative_position_components.png`
- `comparative_position_error.png`
- `comparative_velocity_components.png`
- `comparative_velocity_error.png`
