# Wildlife Detection System - Next Steps and Clarifications

Based on our meetings with Prof. Peeva, we have updated our roadmap and clarified some important aspects of the project.

## Clarifications on Requirements

1. **Diurnal Activity Analysis**
   - Focus on seasonal and annual activity patterns (NOT monthly patterns)
   - Categorize images by light conditions:
     - Full darkness
     - Early twilight
     - Late twilight
     - Daylight
   - Correlate with moon phases and temperature
   - Track snow cover duration (important for ungulates)

2. **Recognition Challenges**
   - System should identify animals from:
     - Partial views of limbs
     - Body parts
     - Ears or distinctive features
   - Particular focus on identifying blurry silhouettes of similar species moving quickly past camera traps (especially for wolf/jackal differentiation)
   - Adapt to varied habitats (plains/Stara Planina mountains)
   - Handle varying lighting conditions
   - Account for different camera positions

3. **Methodological Notes**
   - Focus on the existing database of metadata connecting:
     - Image name
     - Species detected
     - Additional parameters
   - **Important**: For now, we will NOT include specialized data sources like "Mostela" cameras or traps for polecats and weasels

## Timeline and Next Actions

1. **Data Collection**
   - Prof. Peeva will provide primary data and photo materials after April 13, 2025 (following international travel)
   - We will need to allow time for Prof. Peeva to review and prepare the photographic materials

2. **Immediate Development Tasks (April 2025)**
   - Complete annotation of test_01 dataset
   - Improve annotation interface efficiency
   - Implement environmental parameter tracking
   - Begin work on silhouette recognition algorithms
   - Research solutions for similar species differentiation

3. **Medium-term Goals (May-June 2025)**
   - Train preliminary models with current data
   - Integrate new datasets from Prof. Peeva
   - Implement seasonal activity analysis
   - Develop habitat classification components
   - Begin work on behavioral tracking

4. **Long-term Vision (Q3-Q4 2025)**
   - Full integration of environmental parameters
   - Advanced species differentiation
   - Multi-animal tracking in single frames
   - Behavioral pattern recognition
   - Reporting and visualization dashboard

This project is of significant importance for our state's wildlife conservation efforts and will provide valuable insights for environmental protection policies and biodiversity monitoring.