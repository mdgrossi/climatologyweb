---
layout: home
---
# Climatology

This site contains climatologies (highs, lows, and averages) for atmospheric and oceanic variables derived from NOAA CO-OPS weather and tide observations. Current sites are updated daily; archived sites are updated less frequently. This project is inspired by [Brian McNoldy](https://bmcnoldy.earth.miami.edu/) at the [University of Miami](https://welcome.miami.edu), whose long-standing ["Climatology of Virginia Key, FL" site](https://bmcnoldy.earth.miami.edu/vk/) never ceased to provide insightful weather perspectives during my time at the [Rosenstiel School of Marine, Atmospheric, and Earth Science](https://www.earth.miami.edu/).

### Data
<details style="font-size:13px;">
    <summary>
      What are these data?
    </summary>
  <p>The <a href="https://www.noaa.gov">National Oceanographic and Atmospheric Administration (NOAA)</a> <a href="https://oceanservice.noaa.gov/">National Ocean Service (NOS)</a> Center for Operational Oceanographic Products and Services (CO-OPS) operates hundreds of water level observation stations along the United States coasts and Great Lakes. This <a href="https://tidesandcurrents.noaa.gov/nwlon.html">National Water Level Observation Network (NWLON)</a>, part of the <a href="https://ioos.noaa.gov">Integrated Ocean Observing System (IOOS)</a>, provides the data from which official tidal predictions are generated. Most of these observation stations also observe water temperature as well as air temperature, barometric pressure, and wind. All of these data are publically available via the NOAA CO-OPS <a href="https://tidesandcurrents.noaa.gov/">Tides and Currents</a> data portal.</p>
  
  <p>The historical time series vary among sites and environmental parameters. Water level sensors often came first, with weather stations added later. Data collected since circa 1995 are generally available in 6-minute observations; prior to that, observations are hourly. Data inventories are provided for every site: 
    <ul>
        <li> <a href="https://tidesandcurrents.noaa.gov/inventory.html?id=8656483">Beaufort, NC</a> </li>
        <li> <a href="https://tidesandcurrents.noaa.gov/inventory.html?id=8447930">Woods Hole, MA</a> </li>
        <li> <a href="https://tidesandcurrents.noaa.gov/inventory.html?id=8725110">Naples, FL</a> </li>
        <li> <a href="https://tidesandcurrents.noaa.gov/inventory.html?id=8747437">Bay St. Louis, MS</a> </li>
        <li> <a href="https://tidesandcurrents.noaa.gov/inventory.html?id=8723214">Virginia Key, FL</a> </li>
        <li> <a href="https://tidesandcurrents.noaa.gov/inventory.html?id=8557380">Lewes, DE</a> </li>
    </ul>
  Water level sensors are calibrated and the observations verified. None of the other variables are verified and should be used with caution.
    </p>
</details>

### Methodology
<details style="font-size:13px;">
  <summary>
    How are these statistics calculated?
  </summary>
  <p>All data are retrieved from the NOAA CO-OPS <a href="https://tidesandcurrents.noaa.gov/">Tides and Currents</a> data portal. <a href="https://tidesandcurrents.noaa.gov/faq.html">Data query lengths are restricted</a> in order to prevent large data requests from hogging server resources, a limit that affects both web-based retrieval and application programming interace (API) calls. It is therefore necessary to make repeated successive queries in order to download longer time periods of data, a task for which a Python API wrapper called "<a href="https://github.com/GClunies/noaa_coops">noaa-coops</a>" was developed.</p>
      
  <p>To initiate a climatology, the noaa-coops utility is used to download all data from the beginning of the time series through the time of initiation. These historical data are saved to file to avoid having to repeatedly re-download historical data. To update the climatology, either daily, monthly, or on some other interval, data are downloaded starting from the latest time stamp in the saved historical data and appended to the saved data. Six-minute data are used whenever possible and hourly observations otherwise. </p>

  <p>Any data flagged by NOAA as being suspect for any reason (flag > 0) are discarded, for example, minimum or maximum expected values or rate of change tolerance exceeded. A day is allowed to have up to three hours of missing data to be counted, and a month is allowed up to two days of missing data to be counted. Climatological statistics are calculated as follows.</p>
</details>
