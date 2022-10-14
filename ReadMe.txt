This project is the Source Code of Seq-Trans, which is expertised in long time series prediction.

Steps:
1. Set 'config.py', data_name can be 'ETT' or 'Weather'
2. Run 'ETT.py' or 'Weather.py' in 'DataProvider' folder to generate dataset
3. Run 'Train.py' to start training Seq Trans
4. Run 'Test.py' to test the training result, and can see visualization of prediction results

Tips:
1. The cuda version of pytorch is 11.3, but we think 10.2 is OK as well
2. Besides Pytorch, numpy, pandas, sklearn and matplotlib is also needed, as shown in 'requirements.txt'
3. We have used four datasets in paper, but Taxi and ECL dataset are too large, so the package just contains ETT and Weather dataset, so do not run 'Taxi.py' and 'ECL.py'
4. Explanation of parameters have been written in 'config.py' with the same notation in paper.

Pytorch Installation Command: pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

***
In 20th line of 'Test.py', we translate the output prediction sequence to guarantee the first time point of prediction sequence is equal to the last point of historical sequence.
This process has not been mentioned and explained in the paper because it is not a part of Seq-Trans and we really do not have enough space, so we give a brief explanation here.

Explanation:
The prediction ability of Seq-Trans completely comes from learning from fluctiations, so it has few ideas about the absolute position of curve.
Therefore, we use value of the last historical point to tell Seq-Trans a general position of where the prediction should begin.

This process is quite useful if testing data comes from a period of time that is completely not covered by time in training data. For example, the training data is in August but the testing data is in September.
However, if the testing data is randomly selected from the whole datasets, this process may be unnecessary because training data has contained position information of sequence in testing data.


Thank you for your Patience!
2022.08.17