import random
from faker import Faker

cities_usa = ['NewYork','LosAngeles','Chicago','Houston','Phoenix','Philadelphia','SanAntonio','SanDiego','Dallas','SanJose','Miami','Seattle','Denver','Boston','Nashville','Atlanta','LasVegas','Portland','Charlotte','SanFrancisco']
cities_india = ['Lucknow','Kanpur','Nagpur','Patna','Bhopal','Ludhiana','Agra','Nashik','Vadodara','Varanasi','Mumbai','Delhi','Bangalore','Hyderabad','Ahmedabad','Chennai','Kolkata','Surat','Pune','Jaipur']
cities_uk =['London','Birmingham','Manchester','Leeds','Liverpool','Newcastle','Sheffield','Bristol','Coventry','Leicester','Glasgow','Edinburgh','Belfast','Cardiff','Southampton','Plymouth','Brighton','Aberdeen','Norwich','Nottingham']


india_faker = Faker('en_IN')
uk_faker = Faker('en_GB')
us_faker = Faker('en_US')

names = []

# Generate 1000 random names from each country
for i in range(1000):
    names.append(india_faker.name())
    names.append(uk_faker.name())
    names.append(us_faker.name())

# Generate random ages
def generate_age():
    return random.randint(18, 65)

# Generate random addresses
def generate_address():
    streets = ['Main St', 'Oak St', 'Maple Ave', 'Elm St', 'Cedar Ln', 'Pine Rd', 'Spruce St', 'Birch Ave', 'Cherry St', 'Poplar Dr']
    return f"{random.randint(1, 9999)} {random.choice(streets)}"

# Generate City
def generate_cities():
    cities = ['NewYork','LosAngeles','Chicago','Houston','Phoenix','Philadelphia','SanAntonio','SanDiego','Dallas','SanJose','London','Birmingham','Manchester','Leeds','Liverpool','Newcastle','Sheffield','Bristol','Coventry','Leicester','Mumbai','Delhi','Bangalore','Hyderabad','Ahmedabad','Chennai','Kolkata','Surat','Pune','Jaipur','Miami','Seattle','Denver','Boston','Nashville','Atlanta','LasVegas','Portland','Charlotte','SanFrancisco','Glasgow','Edinburgh','Belfast','Cardiff','Southampton','Plymouth','Brighton','Aberdeen','Norwich','Nottingham','Lucknow','Kanpur','Nagpur','Patna','Bhopal','Ludhiana','Agra','Nashik','Vadodara','Varanasi']
    return random.choice(cities)
# Generate State
def generate_states():
    states = ['California','Texas','Florida','NewYork','Pennsylvania','Illinois','Ohio','Georgia','NorthCarolina','Michigan','England','Scotland','Wales','NorthernIreland','UttarPradesh','Maharashtra','Bihar','WestBengal','MadhyaPradesh','TamilNadu','Rajasthan','Karnataka','Gujarat','AndhraPradesh','Arizona','Colorado','Indiana','Iowa','Kansas','Kentucky','Louisiana','Maryland','Minnesota','Missouri','Nebraska','Nevada','NewJersey','Oregon','SouthCarolina','Tennessee','Virginia','Washington','Wisconsin','Kerala','Punjab','Haryana','Telangana','Jharkhand','Chhattisgarh','Uttarakhand','HimachalPradesh','Assam','Odisha']
    return random.choice(states)

def generate_country():
    return random.choice(['United States','United Kingdom','India'])

# Generate a million records
with open('simple_dataset.csv', 'w', encoding='UTF-8') as file:
    file.write('ID,Name,Age,Address,City,State,Country,Postcode\n')
    for i in range(35000000):
        name = random.choice(names)
        age = generate_age()
        address = generate_address()
        city = generate_cities()
        state =  generate_states()
        if(city in cities_usa):
            country = 'United States'
            postcode = us_faker.postcode()
        if(city in cities_uk):
            country = 'United Kingdom'
            postcode = uk_faker.postcode()
        if(city in cities_india):
            country = 'India'
            postcode = india_faker.postcode()

        file.write(f"{i},{name},{age},{address},{city},{state},{country},{postcode}\n")
