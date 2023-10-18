# Aviators-Phase-5-Project-Final

# X (Twitter) Sentiment Analyis for Top Airlines in the USA
<img src="Images\1_UbeX1kOVLtCsJTpMUrWQuw.jpg"/>

## NAMES
1. Joseph Kinuthia 
2. Emily Owiti
3. James Mungai
4. Paul Muriithi
5. Raphael Kariuki
6. Sylvia Muchiri

# TABLE OF CONTENTS
1. [INTRODUCTION](#INTRODUCTION)
2. [BUSINESS UNDERSTANDING](#BUSINESS-UNDERSTANDING)
3. [PROBLEM STATEMENT](#PROBLEM-STATEMENT)
4. [MAIN OBJECTIVE](#MAIN-OBJECTIVE)
5. [SPECIFIC OBJECTIVES](#SPECIFIC-OBJECTIVES)
6. [RESEARCH QUESTIONS](#RESEARCH-QUESTIONS)
7. [EXPERIMENTAL DESIGN](#EXPERIMENTAL-DESIGN)
8. [MODELLING, CHATBOT DEVELOPMENT AND DEPLOYMENT](#MODELLING,-CHATBOT-DEVELOPMENT-AND-DEPLOYMENT)
9. [CONCLUSION](#CONCLUSION)
10. [RECOMMENDATIONS](#RECOMMENDATIONS)

# INTRODUCTION

In this project, we focus on leveraging Natural Language Processing (NLP) techniques to analyze sentiments expressed in a Twitter dataset, specifically within the airline industry domain. The primary objectives include sentiment analysis, the construction of a precise tweet classification model, and the development of a chatbot capable of responding to customer feedback and directing queries to the appropriate resolution teams. By accomplishing these goals, we will create solutions that enable airline companies to extract valuable insights from social media data which will play a big role in enhancing their customer service, improving their service offering and empowering these organizations to make data-driven decisions.
# BUSINESS UNDERSTANDING.

The global airline industry connects people and cargo worldwide, comprising diverse carriers. Key trends include tech adoption, sustainability, personalized travel, partnerships, and post-pandemic safety. Profits vary based on market conditions, fuel prices, competition, and trends, averaging 2-5%. Strong brands reduce customer churn, boosting prices and revenues.

# PROBLEM STATEMENT.

The advent of social media has generated an abundance of data, presenting both opportunities and challenges for organizations. This vast pool of data offers unparalleled insights into customer perceptions, preferences, and feedback. However, many organizations are yet to develop frameworks and strategies to effectively analyze and interpret such data. Insights from this data holds the potential to benefit various domains, including business operations, marketing strategies, public opinion analysis, and more.

Our stakeholders (top American airline companies) have requested us to analyze social media raw data and showcase the customer sentiment as either positive, neutral or negative while identifying the top drivers for these sentiments.

Our dataset is sourced from Twitter, capturing a wide array of tweets, and our primary focus is analyzing and visualizing drivers for key & top public & customer sentiment. We aim to address critical questions and challenges faced by airlines, such as understanding passenger sentiments from unstructured data and predicting engagement metrics. By doing so, we strive to provide airlines with the tools and knowledge needed to enhance customer experiences, optimize operations, and make data-driven decisions in an ever-evolving and competitive industry.

# MAIN OBJECTIVE.
The primary goal of the project is sentiment analysis to analyze raw tweets to extract the public sentiment as well as development of a chatbot.

# SPECIFIC OBJECTIVES.
1. To analyze the data & derive the public's sentiment (positive, neutral or negative) for our client
2. To build a model that can classify raw tweets into the three sentiment classes for future use
3. To visualize the top drivers for each sentiment category to help management target service delivery improvement
4. To create a chatbot to monitor customer feedback on X that provides realtime responses to customers

# RESEARCH QUESTIONS.
1. What are the predominant sentiments expressed by passengers on X regarding major U.S. airlines?
2. What are the most common reasons for negative sentiments among airline passengers, as expressed in their tweets?
3. How does the sentiment compare between the various airlines in our dataset (highest positive, negative, neutral)?

# EXPERIMENTAL DESIGN.
1. Exploratory data analysis & data cleaning (including categorical variable encoding, feature engineering)
2. Data preprocessing for NLP (preparing text for sentiment scoring)
3. Data labelling & determining labelling accuracy
3. Sentiment analysis
4. Visualization of sentiment analysis outcomes
5. Prediction model building & validation
6. Chatbot development
7. Prediction model & chatbot deployment

# MODELLING, CHATBOT DEVELOPMENT AND DEPLOYMENT.

In the context of engagement prediction, the approach involves several key steps: data splitting for training, testing, and validation, followed by the training and evaluation of various regression models. These models are assessed using metrics like MAE, RMSE, R-squared, and cross-validation scores, with an emphasis on achieving high accuracy in sentiment analysis. Hyperparameter tuning and ensemble modeling are used to enhance model performance, and the best-performing models are selected. In parallel, a chatbot is developed with sentiment-based routing, using predefined or dynamic response templates based on sentiment classification. Additionally, natural language understanding (NLU) is implemented to extract context from user tweets for personalized responses. Testing and training ensure appropriate responses to various sentiments, and the chatbot is integrated into the user interface and deployed at scale, using Pickle for model deployment. The project's scalability allows multiple airlines to customize sentiment analysis and engagement prediction for their specific needs, fostering industry-wide improvements in customer engagement and satisfaction.

# CONCLUSION.

**Sentiment Distribution:** The analysis revealed that negative sentiments are most prevalent among passengers, accounting for approximately ***62.69%** of the sentiments in the dataset. This highlights the importance of addressing customer concerns and improving overall satisfaction.

**Airlines' Sentiment Distribution:** Different airlines exhibit varying sentiment distributions, with some struggling with predominantly negative sentiments, while others maintain a more balanced distribution of positive, negative, and neutral sentiments. Understanding these variations is crucial for each airline's strategy.

**Common Negative Reasons:** "Customer Service Issue" is the most frequently cited reason for negative sentiments across airlines, followed by "Late Flight" and "Cancelled Flight." Addressing these common issues can significantly improve passenger satisfaction.

**Model Performance:** The logistic regression and SVM models exhibit strong performance in classifying sentiment, making them suitable choices for implementing a sentiment analysis system for customer interactions.

# RECOMMENDATIONS.

**Improve Customer Service:** Given that "Customer Service Issue" is a predominant reason for negative sentiments across airlines, it's essential for airlines to invest in enhancing their customer service. This includes better training for staff, faster response times, and improved communication with passengers.

**Address Flight Punctuality:** Late flights are a major concern for passengers, and airlines should work on minimizing delays and providing accurate information to travelers. Implementing efficient flight scheduling and contingency plans can help mitigate this issue.

**Invest in Luggage Handling:** Passengers' sentiments are negatively affected by issues related to lost and damaged luggage. Airlines should focus on improving baggage handling processes to reduce such incidents and ensure a smoother travel experience.

**Enhance Online Booking Systems:** Flight booking problems are a common complaint among passengers. Airlines should update and streamline their online booking systems to make it easier and more user-friendly.

**Monitor Social Media:** Airlines should actively monitor social media platforms for customer feedback and respond promptly to address concerns or complaints. This can help improve customer satisfaction and brand reputation.



