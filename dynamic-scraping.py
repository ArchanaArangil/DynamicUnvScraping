import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import sqlite3
import time

driver = webdriver.Chrome()
conn = sqlite3.connect('test-courses.db')
cursor = conn.cursor()
scrapedCourses = False

def check_courses(): #check if page has title courses/subejct and that there is some table or list present
    h1_tags = driver.find_elements(By.TAG_NAME, 'h1')
    for h1 in h1_tags:
        if 'course' in h1.text.lower() or 'subject' in h1.text.lower():
            ul_elements = driver.find_elements(By.TAG_NAME, 'ul')
            table_elements = driver.find_elements(By.TAG_NAME, 'table')
            if ul_elements or table_elements:
                return True
            else:
                return False
    return False

def insert_data(title, description):
    cursor.execute('SELECT 1 FROM course_list WHERE college_name = ? AND course_name = ? AND course_description = ?', ("University of Illinois Urbana-Champaign", title, description))
    exists = cursor.fetchone()
    if not exists:
        cursor.execute('''
            INSERT OR IGNORE INTO course_list (college_name, course_name, course_description) VALUES (?, ?, ?)
        ''', ("University of Illinois Urbana-Champaign", title, description))
        conn.commit()

def scrape1(original_href):
    main_content = driver.find_element(By.CSS_SELECTOR, 'div[id*="Content"], div[id*="content"]')
    ul_elements = main_content.find_elements(By.TAG_NAME, 'ul')
    for ul_index in range(len(ul_elements)):
        main_content = driver.find_element(By.CSS_SELECTOR, 'div[id*="Content"], div[id*="content"]')
        ul_elements = main_content.find_elements(By.TAG_NAME, 'ul')
        ul = ul_elements[ul_index]
        a_list = ul.find_elements(By.TAG_NAME, 'a')
        hrefs = []
        for a in a_list:
            if a.get_attribute('href'):
                print(a.get_attribute('href'))
                if not "#" in a.get_attribute('href'):
                    hrefs.append(a.get_attribute('href'))
        for href in hrefs:
            print(href)
            driver.get(href)
            try:
                main_div = driver.find_element(By.CSS_SELECTOR, '[id*="content"], [id*="Content"]')
                keywords = ['result', 'course', 'Course', 'Result']
                result_divs = []
                divs = main_div.find_elements(By.TAG_NAME, 'div')
                for div in divs:
                    div_class = div.get_attribute('class').lower()
                    if any(keyword in div_class for keyword in keywords):
                        result_divs.append(div)
                for result_div in result_divs:
                    try:
                        result_div.find_element(By.CSS_SELECTOR, '[class*="toggle_btn"]').click()
                    except:
                        pass
                    try:
                        try:
                            title = result_div.find_element(By.CSS_SELECTOR, '[class*="Title"]').text
                            description= result_div.find_element(By.CSS_SELECTOR, '[class*="Desc"]').text
                        except:
                            title = result_div.find_element(By.CSS_SELECTOR, '[class*="title"]').text
                            description= result_div.find_element(By.CSS_SELECTOR, '[class*="desc"]').text                    
                        insert_data(title, description)
                        scrapedCourses = True
                    except:
                        continue
            except Exception as e:
                 print(f"an error occurred: {str(e)}")

            driver.get(original_href)
      
def nav1():
    try:
        url = "https://illinois.edu/"
        driver.get(url)
       
        # Is there a search button
        try:
            driver.find_element(By.CSS_SELECTOR, 'button[class*="-search"], button[class*="search-"]').click()
        except:
            try:
                driver.find_element(By.CSS_SELECTOR, 'button[id*="search"]').click()
            except:
                try:
                    driver.find_element(By.CSS_SELECTOR, 'div[class*="-search"]').click()
                except:
                    try:
                        driver.find_element(By.CSS_SELECTOR, 'a[class*="-search"], button[class*="search-"]').click()
                    except:
                        print("search button not found")
        time.sleep(1)

        # Find the search textbox & perform search
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, 'input[id*="q"]')
        except:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, 'input[id*="search"]')
            except:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, 'input[class*="search"]')
                except:
                    try:
                        elements = driver.find_elements(By.CSS_SELECTOR, 'input[placeholder*="Search"]')
                    except:
                        print("no element found")

        for element in elements:
            try:
                #element.click()
                element.send_keys('course catalog')
                element.send_keys(Keys.RETURN)
                break
            except:
                print("First element did not work")

        # Wait & get results
        time.sleep(3)
        try:
            results = driver.find_elements(By.CSS_SELECTOR, '[id*="results"]')
            links = []
            for result in results:
                links = links + (result.find_elements(By.TAG_NAME, "a"))
        except:
            links = driver.find_elements(By.TAG_NAME, "a")

        filtered_links = []
        for link in links:
            if link.get_attribute("href") is not None:
                if ('course' in link.get_attribute('href')) and not(url in link.get_attribute('href')):
                    filtered_links.append(link.get_attribute('href'))

        for href in filtered_links:
            if not scrapedCourses:
                driver.get(href)
                try:
                    WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.TAG_NAME, 'h1')))
                    page_reached = check_courses()

                    if not page_reached:
                        links = driver.find_elements(By.XPATH, "//a[contains(@title, 'course') or contains(text(), 'course')]")
                        for link in links:
                            href = link.get_attribute('href')
                            driver.get(href)
                            time.sleep(3)
                            if check_courses():
                                return
                    else:
                        scrape1(href)
                except:
                    print("New page did not load properly.")
                driver.back()
            

    except Exception as e:
        print(f"an error occurred: {str(e)}")


nav1()
