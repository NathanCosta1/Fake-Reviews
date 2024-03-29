import requests
from bs4 import BeautifulSoup
import csv

movie_urls = [
    "https://letterboxd.com/film/poor-things-2023/reviews/by/activity/page/{}/",
    "https://letterboxd.com/film/dune-2021/reviews/by/activity/page/{}/",
    "https://letterboxd.com/film/anyone-but-you/reviews/by/activity/page/{}/",
    "https://letterboxd.com/film/american-fiction/reviews/by/activity/page/{}/",
    "https://letterboxd.com/film/the-marvels/reviews/by/activity/page/{}/",
    "https://letterboxd.com/film/past-lives/reviews/by/activity/page/{}/",
    "https://letterboxd.com/film/the-zone-of-interest/reviews/by/activity/page/{}/",
    "https://letterboxd.com/film/upgraded/reviews/by/activity/page/{}/",
    "https://letterboxd.com/film/society-of-the-snow/reviews/by/activity/page/{}/",
    "https://letterboxd.com/film/the-iron-claw-2023/reviews/by/activity/page/{}/"
]

# Create CSV to store review data
with open('C:/Users/82nat/OneDrive/Desktop/reviews_data.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # Create header 
    writer.writerow(["Review Text", "Star Rating", "Total Films Reviewed", "Reviews This Year", "Following", "Followers"])
    
    for url in movie_urls:
        page_number = 1
        while True:
            try:
                # Construct the full URL for the current page
                full_url = url.format(page_number)
                response = requests.get(full_url)
                response.raise_for_status()  # Raise an exception for HTTP errors
                
                soup = BeautifulSoup(response.content, 'html.parser')
                reviews = soup.find_all("div", class_="body-text -prose collapsible-text")
                stars = soup.find_all("span", class_="rating")
                avatars = soup.find_all("a", class_="avatar -a40")
                
                if not reviews:  # Last page reached
                    break
                
                for review, star, avatar in zip(reviews, stars, avatars):
                    review_text = review.get_text(strip=True)
                    star_rating = star.get("class")[-1][-1]  # Extract the last character of the class (there is other extraneous data in this class)
                    user_profile_link = "https://letterboxd.com" + avatar["href"] 
                    
                    # Go to User's profile through the link in their avatar - this will give us access to relevant user data 
                    profile_response = requests.get(user_profile_link)
                    profile_response.raise_for_status()  # Raise an exception for HTTP errors
                    
                    profile_soup = BeautifulSoup(profile_response.content, "html.parser")
                    total_films_reviewed = profile_soup.find("span", class_="value").get_text(strip=True).replace(",", "")
                    reviews_this_year = profile_soup.find_all("span", class_="value")[1].get_text(strip=True).replace(",", "")
                    following_tag = profile_soup.find("a", class_="thousands")

                    try:
                        following = profile_soup.find_all("span", class_="value")[3].get_text(strip=True).replace(",", "")
                    except:
                        following = "0"

                    followers_tag = profile_soup.find("a", class_="thousands")
                    if followers_tag:
                        followers = followers_tag.find("span", class_="value").get_text(strip=True).replace(",", "")
                    else:
                        followers = "0"
                  
                    if "this review may contain spoilers" not in review_text.lower(): 
                        # Write current review and all data to a row in CSV file
                        writer.writerow([review_text, star_rating, total_films_reviewed, reviews_this_year, following, followers])

            except Exception as e:
                print(f"Error: {e}")
            
            finally:
                page_number += 1
