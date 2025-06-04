These files have static hardcoded values because they run in node71. 

To collect data spectrograms and KPIs as images struct synchronized: 
-----------------------------------------------------------------------
1) Specify configuration options for data collection in the config file
2) run the application_kpi_collector file 
3) run the phy_kpis_collector file
4) run the spectrograms_collector file
-----------------------------------------------------------------------

Synchronize and normalize the data
-----------------------------------------------------------------------
1) Run the sync_and_norm_kpis file
-----------------------------------------------------------------------

Convert time-series KPI to images (224x224)
-----------------------------------------------------------------------
2) Run the creating_kpi_images file
-----------------------------------------------------------------------

After all the collection of the spectrograms convert them to square images (not 224x224 now), the augmenter will do it (224x224)
Only the time series images are 224x224
-----------------------------------------------------------------------
3) Run the preprocess_spectrograms file
-----------------------------------------------------------------------

