import random


def generate_historic_sale(fair_value, mileage_penalty, damage_adjustment):
    locations = ["Miami, FL", "Atlanta, GA", "Austin, TX", "Columbus, OH",
                 "Denver, CO", "Phoenix, AZ", "Raleigh, NC", "Nashville, TN"]
    damage_notes = list(damage_adjustment.keys())
    # weighted so the AVERAGE damage adjustment across categories nets to ~$0 —
    # otherwise 5 negative categories vs 1 positive category skews comp prices
    # below fair_value even though each individual range looks "reasonable"
    damage_weights = [0.45, 0.11, 0.11, 0.11, 0.11, 0.11]

    mileage = random.randint(55000, 95000)
    notes = random.choices(damage_notes, weights=damage_weights, k=1)[0]
    prior_owners = random.choice([1, 1, 1, 2, 2, 3])

    price = fair_value
    price -= (mileage - 75000) * mileage_penalty
    price += damage_adjustment[notes]
    price += random.gauss(0, 200)  # residual noise: unmodeled factors (timing, buyer luck, etc.)

    return {
        "price": round(price),
        "mileage": mileage,
        "days_since_sale": random.randint(3, 90),
        "prior_owners": prior_owners,
        "location": random.choice(locations),
        "notes": notes,
    }


def generate_scenario():
    # randomized PER SCENARIO (not per comp) so all 10 comps AND the subject
    # car itself are priced by the same internally-consistent rule, but the
    # exact price-condition relationship isn't a fixed formula an agent could
    # learn/exploit across many episodes
    mileage_penalty = random.uniform(0.03, 0.08)  # $ per mile above 75k baseline
    damage_adjustment = {
        "clean, no damage": random.randint(200, 400),
        "minor scratches": random.randint(-200, -50),
        "small dent rear bumper": random.randint(-500, -250),
        "windshield chip repaired": random.randint(-300, -100),
        "curb rash on wheels": random.randint(-250, -75),
        "repainted hood": random.randint(-450, -250),
    }
    damage_notes = list(damage_adjustment.keys())
    damage_weights = [0.45, 0.11, 0.11, 0.11, 0.11, 0.11]  # see generate_historic_sale

    # --- Generate the SUBJECT car's own condition first ---
    # (previously this was independent random noise, disconnected from
    # fair_value and using a totally different damage vocabulary than the
    # comps — meaning the car's own price wasn't actually explained by its
    # own condition. Now it uses the exact same categories and coefficients
    # as the comps, so an agent citing "my car has X miles/condition" is
    # citing something that actually determines this car's own fair_value.)
    car_mileage = random.randint(70000, 82000)
    car_damage = random.choices(damage_notes, weights=damage_weights, k=1)[0]

    # baseline value of a "clean, 75k-mile" example of this model/year —
    # this replaces the old flat fair_value range as the new starting point
    baseline_value = random.randint(12500, 14000)

    # the ACTUAL fair_value for THIS specific car, adjusted for its own
    # mileage and condition using the same rule the comps are priced by
    fair_value = round(
        baseline_value
        - (car_mileage - 75000) * mileage_penalty
        + damage_adjustment[car_damage]
    )

    # half-width of the bargaining zone. Symmetric around fair_value so
    # neither reservation is "objectively" favored by where fair_value sits.
    half_zopa = 375
    seller_reservation = fair_value - half_zopa
    buyer_reservation = fair_value + half_zopa  # guaranteed > seller_reservation, always

    historic_sales = [
        generate_historic_sale(fair_value, mileage_penalty, damage_adjustment)
        for _ in range(10)
    ]

    # anchored off fair_value directly (not off the noisy comps' max), so a
    # lucky high comp can't blow the opening anchor sky-high
    asking_price = round(fair_value * 1.15)
    buyer_opening_anchor = round(fair_value * 0.85)

    return {
        "car_facts": {
            "model": "Honda Civic EX",
            "year": 2018,
            "mileage": car_mileage,
            "color": random.choice(["silver", "black", "blue", "white", "gray"]),
            "damage": car_damage,
        },
        "days_on_market": random.randint(5, 45),
        "historic_sales": historic_sales,
        "seller_reservation": seller_reservation,
        "buyer_reservation": buyer_reservation,
        "fair_value": fair_value
        #"asking_price": asking_price,
        #"buyer_opening_anchor": buyer_opening_anchor,
        
    
    }