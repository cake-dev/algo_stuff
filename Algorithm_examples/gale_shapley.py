# an implementation of the gale shapley algorithm
# for the stable marriage problem

def gale_shapley(men_prefs, women_prefs):
    # Initialize empty engagement list and free men list
    engagements = {}
    free_men = list(men_prefs.keys())
    
    while free_men:  # While there are free men
        # Choose a free man 'm'
        m = free_men[0]
        
        # Get m's preference list
        m_prefs = men_prefs[m]
        
        # Loop through m's preference list
        for w in m_prefs:
            # If m has already proposed to all, break
            if not m_prefs:
                break
            
            # Pop the highest-ranked woman 'w' to whom m has not yet proposed
            w = m_prefs.pop(0)
            
            # If w is free
            if w not in engagements:
                engagements[w] = m  # (m, w) become engaged
                free_men.remove(m)  # Remove m from free men list
                break
            
            # If w is already engaged to m'
            else:
                m_prime = engagements[w]
                
                # If w prefers m' (her current engagement) to m
                if women_prefs[w].index(m_prime) < women_prefs[w].index(m):
                    continue  # m remains free
                
                # Else w prefers m to m'
                else:
                    engagements[w] = m  # (m, w) become engaged
                    free_men.remove(m)  # Remove m from free men list
                    free_men.append(m_prime)  # m' becomes free
                    break
    
    return engagements  # Return the set of engaged pairs

# Example preference lists
men_prefs = {
    'm1': ['w1', 'w2', 'w3'],
    'm2': ['w2', 'w1', 'w3'],
    'm3': ['w3', 'w2', 'w1']
}

women_prefs = {
    'w1': ['m1', 'm2', 'm3'],
    'w2': ['m2', 'm1', 'm3'],
    'w3': ['m3', 'm1', 'm2']
}

# Run the Gale-Shapley algorithm
result = gale_shapley(men_prefs, women_prefs)
print("Engaged pairs:", result)

# Complexity Upper Bound : O(n^2)