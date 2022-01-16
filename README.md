# ProductRecommendation

## An e-commerce company has captured a huge market share in many fields, and it sells the products in various categories. To become a major leader in the market, it has to compete with the Market Leaders.
The concept of recommendations has been around from the day the Market Leaders added “top picks for you” option in their website.

1. Findings show that 40% of consumers have purchased something more expensive than they had originally set out for. Moreover, nearly half (49%) of the shoppers made an impulse purchase after they received a personalised recommendation and a majority (85%) of them were very happy with their decision.
2. It would be impossible to keep recommendations relevant to each customer in real-time without using a scalable data science model that works well for your business. Data science helps automate customer recommendations to not only make them relevant to each user but also to consider the dynamic changes that occur along the way.
Engagement with Product Recommendations in e-commerce can generate recommendations unique to each user.

For our Analysis, we have Data with 30000 reviews on different products collected.

The steps to be performed for the recommendation are:

    >>  Exploratory data analysis
    >>  Data cleaning
    >>  Text preprocessing
    >>  Feature extraction
    >>  Training a text classification model
    >>  Building a recommendation system
    >>  Improving the recommendations using the sentiment analysis model
    
**Collaborative Filtering:**
The most popular of all product recommender methods, the collaborative filtering technique relies solely on how other customers and users have previously rated a product they purchased.
Idea: 
    If a person X likes items 1, 2, 3 and Y like 2, 3, 4 then they have similar interests and X should like item 4 and Y should like item 1.
    This idea made this algorithm to be most popular.

Basic assumptions:

    >>  Customers who had similar tastes in the past, will have similar tastes in the future
    >>  Users give ratings to catalog items 
    
**User-User collaborative filtering**
A better approach to this method is to take the large database of user reviews and find the user pattern.

User-User Filter:

    >>  User A rates a product with 4 stars.
    >>  B rates the same product with 4 stars.
    >>  User A then likes a product and gives 5 stars to it.
    >>  B will be recommended the same product that user A has rated with 5 stars since it assumes that user B will also like it.
    >>  …
    
**Item-Item Filter:**

With the item-to-item personalization method, products within a single user’s profile are interconnected without relying on other shoppers.
For instance, if the user previously bought cookware, then the user will be recommended with products related to cooking items.


Based on the above mentioned concepts, i have implemented the Recommendation model for the dataset.
I have built User based recommendation system using 3 models listed below:


  >> SVD AutoBuildTestset Model

      >- SVD algorithm is equivalent to Probabilistic Matrix Factorization. 
      It will Automatically generate the testdata to evaulate the model. No need for test-train split
  >> SVD Train Test Split
 
      >- In the SVD model, we will split the same data as Train and Test split and train the model to get the user recommendation.
  >> User rating using pivot table

      >- Creating pivot for the user and Items with the ratings as values which helps in finding the recommendation.

Using RMSE value, i have found the best model and built the user recommendation.

For the Item Based Recommendation, I have used:

  >> KNNWithMeans

      >- Creates similarity matrix which provides the nearest neighbours
  >> NearestNeighbors
 
      >- Unsupervised model which helps in identifying the clusters of similar products and recommend product


**Deployment**:

1. Created Requirements.txt file for all the software requirements
2. Creat Procfile and deploy in Heroku

**Application URL**:
