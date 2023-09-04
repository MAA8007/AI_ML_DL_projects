
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer

# Load the saved model
loaded_model = tf.keras.models.load_model('news classifier/news_classifier_optimized')


# Define the label_binarizer for inverse transform. Ideally, you should save this from training and load it here.
label_binarizer = LabelBinarizer()
label_binarizer.fit(['U.S. NEWS', 'COMEDY', 'PARENTING', 'WORLD NEWS', 'CULTURE & ARTS', 'TECH',
 'SPORTS', 'ENTERTAINMENT', 'POLITICS', 'WEIRD NEWS', 'ENVIRONMENT',
 'EDUCATION', 'CRIME', 'SCIENCE', 'WELLNESS', 'BUSINESS', 'STYLE & BEAUTY',
 'FOOD & DRINK', 'MEDIA', 'QUEER VOICES', 'HOME & LIVING', 'WOMEN',
 'BLACK VOICES', 'TRAVEL', 'MONEY', 'RELIGION', 'LATINO VOICES', 'IMPACT',
 'WEDDINGS', 'COLLEGE', 'PARENTS', 'ARTS & CULTURE', 'STYLE', 'GREEN', 'TASTE',
 'HEALTHY LIVING', 'THE WORLDPOST', 'GOOD NEWS', 'WORLDPOST', 'FIFTY', 'ARTS',
 'DIVORCE'])

# Disclaimer: The following data is completely made up. 
test_data = [
    {'headline': 'Apple finally reveals its long awaited phone', 'category': 'TECH'},
    {'headline': 'UFOs: Five revelations from Nasas public meeting', 'category': 'SCIENCE'},
    {'headline': 'Joe Biden has defeated his long standing Republican rival, Trump', 'category': 'POLITICS'},
    {'headline': 'The GDP of Pakistan has fallen by 1%', 'category': 'BUSINESS'},
    {'headline': 'Don Draper voted the best character in MadMen', 'category': 'ENTERTAINMENT'},
    {'headline': "Fitbit's new tracker promises to get you in shape", 'category': 'WELLNESS'},
    {'headline': 'Knife attacks have once agin risen in London', 'category': 'CRIME'},
    {'headline': 'Copenhagen, the city that is turning trash into electricity', 'category': 'ENVIRONMENT'},
    {'headline': 'Tomato Basil Soup recipe takes the internet by storm', 'category': 'FOOD & DRINK'},
    {'headline': 'High school students to receive free college tuition', 'category': 'EDUCATION'},
    {'headline': 'UN announces new global environmental initiative', 'category': 'WORLD NEWS'},
    {'headline': 'Man claims to have seen Bigfoot, presents blurry photo', 'category': 'WEIRD NEWS'},
    {'headline': 'New solar panel technology breakthrough', 'category': 'GREEN'},
    {'headline': 'Academy announces new diversity requirements for Oscar eligibility', 'category': 'ARTS & CULTURE'},
    {'headline': 'Survey shows Millennials value experiences over possessions', 'category': 'MONEY'},  {'headline': 'Tomato Basil Soup recipe takes the internet by storm', 'category': 'FOOD & DRINK'},
    {'headline': 'High school students to receive free college tuition', 'category': 'EDUCATION'},
    {'headline': 'UN announces new global environmental initiative', 'category': 'WORLD NEWS'},
    {'headline': 'Man claims to have seen Bigfoot, presents blurry photo', 'category': 'WEIRD NEWS'},
    {'headline': 'New solar panel technology breakthrough', 'category': 'GREEN'},
    {'headline': 'Academy announces new diversity requirements for Oscar eligibility', 'category': 'ARTS & CULTURE'},
    {'headline': 'Survey shows Millennials value experiences over possessions', 'category': 'MONEY'},
    {'headline': 'Local comedy club holds annual stand-up competition', 'category': 'COMEDY'},
    {'headline': 'Parents rally for improved school safety measures', 'category': 'PARENTING'},
    {'headline': 'New art exhibit highlights the beauty of imperfection', 'category': 'CULTURE & ARTS'},
    {'headline': 'Game-winning touchdown leads to Super Bowl victory', 'category': 'SPORTS'},
    {'headline': 'TV host announces retirement after 30 years on air', 'category': 'ENTERTAINMENT'},
    {'headline': 'Study reveals the benefits of a good night’s sleep', 'category': 'WELLNESS'},
    {'headline': 'Tech giants collaborate on new cybersecurity measures', 'category': 'TECH'},
    {'headline': 'Music festival lineup announced, features top artists', 'category': 'ENTERTAINMENT'},
    {'headline': 'Local library starts book club for seniors', 'category': 'ARTS & CULTURE'},
    {'headline': 'Economists debate causes of recent stock market fluctuations', 'category': 'BUSINESS'},
    {'headline': 'Scientists discover potential cure for common cold', 'category': 'SCIENCE'},
    {'headline': 'Actress advocates for mental health awareness', 'category': 'ENTERTAINMENT'},
    {'headline': 'National parks face overcrowding during holiday season', 'category': 'TRAVEL'},
    {'headline': 'Police solve decades-old cold case', 'category': 'CRIME'},
    {'headline': 'Dietitian shares healthy alternatives to fast food', 'category': 'HEALTHY LIVING'},
    {'headline': 'Newlywed couple shares tips for planning a wedding on a budget', 'category': 'WEDDINGS'},
    {'headline': 'Documentary explores the life and career of jazz legend', 'category': 'ARTS & CULTURE'},
    {'headline': 'Company recalls products due to safety concerns', 'category': 'BUSINESS'},
    {'headline': 'Historic peace treaty signed between rival nations', 'category': 'WORLD NEWS'},
    {'headline': 'Film festival showcases independent cinema', 'category': 'ENTERTAINMENT'},
    {'headline': 'Interview with award-winning author of children’s books', 'category': 'ARTS & CULTURE'},
    {'headline': 'Guide to the best summer destinations for families', 'category': 'TRAVEL'},
    {'headline': 'Beauty expert reviews the latest makeup trends', 'category': 'STYLE & BEAUTY'},
    {'headline': 'International soccer teams compete for championship title', 'category': 'SPORTS'},
    {'headline': 'Entrepreneur shares journey of building a successful business', 'category': 'BUSINESS'},
    {'headline': 'Animal shelter launches adoption event', 'category': 'GOOD NEWS'},
    {'headline': 'Diverse voices in literature gain recognition', 'category': 'ARTS'},
    {'headline': 'Groundbreaking ceremony marks start of community project', 'category': 'IMPACT'},
    {'headline': 'Virtual reality changing the landscape of video games', 'category': 'TECH'},
    {'headline': 'Local farmers market supports sustainability', 'category': 'GREEN'},
    {'headline': 'Parenting in the digital age poses new challenges', 'category': 'PARENTING'},
    {'headline': 'Top five apps to boost productivity', 'category': 'TECH'},
    {'headline': 'New restaurant brings fusion cuisine to the city', 'category': 'FOOD & DRINK'},
    {'headline': 'Investigative report exposes political corruption', 'category': 'POLITICS'},
    {'headline': 'Fitness trends to look out for this year', 'category': 'WELLNESS'},
    {'headline': 'Youth orchestra delivers stunning performance', 'category': 'ARTS & CULTURE'},
    {'headline': 'World leaders gather for climate change summit', 'category': 'ENVIRONMENT'},
  {'headline': 'King Elvis Dead', 'category': 'ENTERTAINMENT'},
    {'headline': 'Greatest Crash in Wall Street’s History', 'category': 'BUSINESS'},
    {'headline': 'Young Elected City’s 1st Black Mayor', 'category': 'POLITICS'},
    {'headline': 'Dewey Defeats Truman', 'category': 'POLITICS'},
    {'headline': 'So What the Hell Happens Now?', 'category': 'POLITICS'},
    {'headline': 'Heir to Austria’s Throne Is Slain with His Wife by a Bosnian Youth to Avenge Seizure of His Country', 'category': 'WORLD'},
    {'headline': 'War! Oahu Bombed by Japanese Planes', 'category': 'WORLD'},
    {'headline': 'Time to Face the Past', 'category': 'SOCIETY'},
    {'headline': 'Beatle John Lennon Slain', 'category': 'ENTERTAINMENT'},
    {'headline': 'Martin King Shot to Death', 'category': 'SOCIETY'},
    {'headline': 'Hitler Dead', 'category': 'WORLD'},
    {'headline': 'Nixon Resigns', 'category': 'POLITICS'},
    {'headline': 'Kennedy Is Killed By Sniper As He Rides In Car In Dallas; Johnson Sworn In On Plane', 'category': 'POLITICS'},
    {'headline': 'Mr. President', 'category': 'POLITICS'},
    {'headline': 'Diana Dead', 'category': 'WORLD'},
    {'headline': 'Mandela Goes Free Today', 'category': 'WORLD'},
    {'headline': 'War on America', 'category': 'WORLD'},
    {'headline': 'The First Footstep', 'category': 'SCIENCE'},
    {'headline': 'Titanic Sinks Four Hours After Hitting Iceberg', 'category': 'WORLD'},
    {'headline': 'PEACE!', 'category': 'WORLD'}
]
    


# Convert the test data to DataFrame
import pandas as pd
df_test = pd.DataFrame(test_data)

# Combine headline and short_description if required
# df_test['text'] = df_test['headline'] + ' ' + df_test['short_description']

# Extract the text data for prediction
X_test = df_test['headline']

# Predict using the loaded model
predictions = loaded_model.predict(X_test)

# Inverse transform to get labels
predicted_labels = label_binarizer.inverse_transform(predictions)

# Output the results
for headline, predicted_label in zip(df_test['headline'], predicted_labels):
    print(f'Headline: {headline} -> Predicted Category: {predicted_label}')

