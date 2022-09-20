# AI4VN_2022-Air_Quality_Forecasting_Challenge_Private

- **Copy data**: You copy data to 2 folders:
    - Data train: './train/air' and './train/meteo'
    - Data public-test: './data_test/input/'
- **Train**:
    ```
    !pip install -r requirements.txt
    ```
    ```
    python train.py --conf_data=./CONFIG/model_thien.yml --name_model=thien1892/
    ```
- **Submit**:
    ```
    python submit.py --path_save_submit=./submit/ --path_data_test=./data-test/input/ --conf_model=./save_model/thien1892/model_save.yml
    ```
- **Submit best score**:
    ```
    python submit_combine2_model.py
    ```
- Submit file **'./submitthien1892.zip'**, **'submitcombine2model.zip'**