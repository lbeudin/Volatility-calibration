# Volatility-calibration
This project was a research project done during my final year engineering. The overall purpose is to calibrate a volatility surface with differente algorithm and pricing model.


Part 1 - Interpolation

From a table of 10 strikes and 10 prices, I interpolate the data to obtain a surface of prices. 
I then use this interpolation to calculate the continuous measure of Breeden-Litzenberger.


Part 2 - Calibration of volatility surfaces.

From the step 1, I obtained an implied volatility surface with newton raphston algorithm with BS formula.  
I compare monte carlo pricing method and black and scholes. 
I calibrate my own pricing surface with monte carlowith Nelder Mead algorithm. 
I improve the calibration by calibrating each parameters vol, vol of vol and hurst exponant. 
As it is hard to calibrate the all surface due to long time computation, I look at a specific point strike 98.3 and maturity = 7 months. 
I also implement a monte carlo with 5 parameters and the Ornstein_Uhlenbeck process. 


Part 3 - Recuit Simulé

I compare the peformance of another calibration algorithm : recuit simulé with neldermead.
