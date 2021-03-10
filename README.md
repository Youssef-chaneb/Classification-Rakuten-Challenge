# Classification-Rakuten-Challenge

2021 10th March

You will find in this repo my contribution to the Multimodal Product Data Classification organized by Rakuten Institute of Technology and ENS. This work has been made in collaboration with Kaouther Bouhlel (@Kaouther0675)

You can find the details of the challenge and the DataSet I used for this project in this link : https://challengedata.ens.fr/challenges/35

The goal of this challenge was to predict the category of pa product with its related image, title & description.

I've created two distincts models to predict the the prodcts' category : the first one is a Resnet based model and ony use images data. The second one is a  one dimension convolutionnal neural network and uses the the title & the description of the product.

We also have integrated those models in two streamlit application :
- The first app enables you to upload a product's image in order to predict its category
- The second app enables you to enter the title & description of the product to predict its category
