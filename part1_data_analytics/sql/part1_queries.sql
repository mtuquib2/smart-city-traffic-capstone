-- Part 1 SQL analysis (extracted from metro_traffic.sqbpro, DB Browser for SQLite)
-- Database: metro_traffic.db, table Metro_Interstate_Traffic_Volume

-- ============================================================================
-- Data Upload Completeness Check
-- ============================================================================
SELECT COUNT(*) FROM "Metro_Interstate_Traffic_Volume"

-- ============================================================================
-- Annual Traffic Trends
-- ============================================================================
SELECT

    SUBSTR(date_time, 1, 4) AS year,

    SUM(traffic_volume) AS total_traffic

FROM Metro_Interstate_Traffic_Volume

WHERE SUBSTR(date_time, 1, 4) IN ('2012','2013','2014','2015','2016','2017')

GROUP BY year

ORDER BY year;

WITH yearly AS (

    SELECT

        SUBSTR(date_time, 1, 4) AS year,

        SUM(traffic_volume) AS total_traffic

    FROM Metro_Interstate_Traffic_Volume

    WHERE SUBSTR(date_time, 1, 4) BETWEEN '2012' AND '2017'

    GROUP BY year

)

SELECT

    y1.year,

    y1.total_traffic,

    y1.total_traffic - y0.total_traffic AS change_from_prev

FROM yearly y1

LEFT JOIN yearly y0

    ON CAST(y1.year AS INTEGER) = CAST(y0.year AS INTEGER) + 1

ORDER BY y1.year;



/* 	

Identify years with increases or decreases. 

2013, 2016 and 2017 saw traffic volume increases

2014 and 2015 saw traffic volume decreases



Calculate the change between years where appropriate. 

Traffic Volume Increases: 2013 (19,968,645); 2016 (15,313,615) and 2017 (5,933,335)

Traffic Volume Decreases:2014 (-12,446,123) and 2015 ( -1,550,083)



Explain at least two meaningful observations about traffic volume trends. 

o	Traffic volume increased in 2013 to ~3.4x 2012 before stabilizing to ~15m average for 2014 and 2015

o	Traffic volume further increased in 2016 and 2017 to 29.5m and 35.4m respectively

*/

-- ============================================================================
-- New Years Day Temperature
-- ============================================================================
SELECT

    SUBSTR(date_time, 1, 4) AS year,

    AVG(temp) AS avg_temp,

    AVG(traffic_volume) AS avg_traffic

FROM Metro_Interstate_Traffic_Volume

WHERE holiday = 'New Years Day'

  AND SUBSTR(date_time, 1, 4) IN ('2015','2016','2017')

GROUP BY year

ORDER BY year;



/* 	

Identify notable year-on-year changes and discuss whether these changes appear relevant to traffic conditions

There is no notably year-on-year changes in temperature on New Year's Day that appear relevant to traffic conditions.

*/

-- ============================================================================
-- Labor Day Temperature
-- ============================================================================
SELECT

    SUBSTR(date_time, 1, 4) AS year,

    AVG(temp) AS avg_temp,

    AVG(traffic_volume) AS avg_traffic

FROM Metro_Interstate_Traffic_Volume

WHERE holiday = 'Labor Day'

  AND SUBSTR(date_time, 1, 4) IN ('2015','2016','2017')

GROUP BY year

ORDER BY year;



/* 	

Identify notable year-on-year changes and discuss whether these changes appear relevant to traffic conditions

There is no notably year-on-year changes in temperature on Labor Day that appear relevant to traffic conditions.

*/

-- ============================================================================
-- Traffic Volume Correlation Analysis
-- ============================================================================
SELECT

    (

        AVG(temp * traffic_volume)

        - AVG(temp) * AVG(traffic_volume)

    )

    /

    (

        SQRT(

            (AVG(temp * temp) - AVG(temp) * AVG(temp))

            *

            (AVG(traffic_volume * traffic_volume) - AVG(traffic_volume) * AVG(traffic_volume))

        )

    ) AS correlation_coefficient

FROM Metro_Interstate_Traffic_Volume;





/* 	

Then:

Interpret the direction of the relationship. 

- There is a positive correlation - as temperature increase, traffic volumen  tends to increase

Interpret the strength of the relationship. 

- Given the correlation is only 0.13 - the relationship is weak positive correlation 

Explain why correlation does not necessarily imply causation.

- Correlation only shows association, not cause-and-effect

- Temperature does not directly cause traffic volume to change, both variables may be influenced by other factors

e.g. time of day, season, holidays, work schedules, work conditions, etc



*/

-- ============================================================================
-- Probabilty and Congestion Analysis
-- ============================================================================
SELECT

    /* Basic probabilities */

    1.0 * SUM(CASE WHEN traffic_volume > 5500 THEN 1 ELSE 0 END) / COUNT(*) 

        AS P_Congestion,



    1.0 * SUM(CASE WHEN weather_main = 'Clear' THEN 1 ELSE 0 END) / COUNT(*) 

        AS P_ClearWeather,



    1.0 * SUM(CASE WHEN traffic_volume > 5500 AND weather_main = 'Clear' THEN 1 ELSE 0 END) / COUNT(*) 

        AS P_Congestion_AND_Clear,



    /* Conditional probabilities */

    1.0 * SUM(CASE WHEN traffic_volume > 5500 AND weather_main = 'Clear' THEN 1 ELSE 0 END)

        / SUM(CASE WHEN traffic_volume > 5500 THEN 1 ELSE 0 END)

        AS P_ClearWeather_given_Congestion,



    1.0 * SUM(CASE WHEN traffic_volume > 5500 AND temp > 292 THEN 1 ELSE 0 END)

        / SUM(CASE WHEN traffic_volume > 5500 THEN 1 ELSE 0 END)

        AS P_HighTemp_given_Congestion,



    /* Odds for clear weather */

    (1.0 * SUM(CASE WHEN traffic_volume > 5500 AND weather_main = 'Clear' THEN 1 ELSE 0 END)

        / SUM(CASE WHEN weather_main = 'Clear' THEN 1 ELSE 0 END))

        /

    (1.0 - (1.0 * SUM(CASE WHEN traffic_volume > 5500 AND weather_main = 'Clear' THEN 1 ELSE 0 END)

        / SUM(CASE WHEN weather_main = 'Clear' THEN 1 ELSE 0 END)))

        AS Odds_Congestion_Clear,



    /* Odds for cloudy weather */

    (1.0 * SUM(CASE WHEN traffic_volume > 5500 AND weather_main = 'Clouds' THEN 1 ELSE 0 END)

        / SUM(CASE WHEN weather_main = 'Clouds' THEN 1 ELSE 0 END))

        /

    (1.0 - (1.0 * SUM(CASE WHEN traffic_volume > 5500 AND weather_main = 'Clouds' THEN 1 ELSE 0 END)

        / SUM(CASE WHEN weather_main = 'Clouds' THEN 1 ELSE 0 END)))

        AS Odds_Congestion_Clouds,



    /* Odds ratio */

    (

        (1.0 * SUM(CASE WHEN traffic_volume > 5500 AND weather_main = 'Clear' THEN 1 ELSE 0 END)

            / SUM(CASE WHEN weather_main = 'Clear' THEN 1 ELSE 0 END))

        /

        (1.0 - (1.0 * SUM(CASE WHEN traffic_volume > 5500 AND weather_main = 'Clear' THEN 1 ELSE 0 END)

            / SUM(CASE WHEN weather_main = 'Clear' THEN 1 ELSE 0 END)))

    )

    /

    (

        (1.0 * SUM(CASE WHEN traffic_volume > 5500 AND weather_main = 'Clouds' THEN 1 ELSE 0 END)

            / SUM(CASE WHEN weather_main = 'Clouds' THEN 1 ELSE 0 END))

        /

        (1.0 - (1.0 * SUM(CASE WHEN traffic_volume > 5500 AND weather_main = 'Clouds' THEN 1 ELSE 0 END)

            / SUM(CASE WHEN weather_main = 'Clouds' THEN 1 ELSE 0 END)))

    )

    AS OddsRatio_Clear_vs_Clouds



FROM Metro_Interstate_Traffic_Volume;





/* 	

Conclude with a short explanation of what the probability analysis suggests about the relationship between weather and congestion.

- Congestion is relatively uncommon [Only 14.7% of all observations exceed 5,500 vehicles]

- Clear weather is not strongly associated with congestion 

    - Only 3.66% of all observations show congestion during clear weather] .

    - Conditional probability P(Clear | Congestion) = 0.2483 means that when congestion occurs, only about 1 in 4 cases happen under clear skies.

- High temperature is not a strong driver of congestion

    - P(High Temp | Congestion) = 0.2630 [Only about 26% of congestion events occur when temperature exceeds 292]

- Odds ratio shows congestion is less likely in clear weather

     - Odds of congestion in clear weather: 0.1516

     - Odds of congestion in cloudy weather: 0.2062

     - Odds ratio = 0.7354 (< 1)

   This means congestion is about 26% less likely in clear weather compared to cloudy weather.

Independence check

Compare:

- P(Congestion AND Clear) = 0.0366

- P(Congestion) × P(Clear) = 0.1473 × 0.2778 ≈ 0.0409

These values are close, suggesting weather and congestion are approximately independent.



Final Conclusion

The probability analysis suggests that congestion is relatively infrequent and only weakly related to weather conditions. 

Clear weather does not significantly increase the likelihood of congestion, as shown by the low joint probability (0.0366) and a conditional probability of only 0.2483. 

High temperatures also show a weak association with congestion. The independence check shows that 𝑃(𝐴∩𝐵) is close to 𝑃(𝐴) 𝑃(𝐵),indicating that congestion and weather behave largely independently. 

The odds ratio (0.7354) further confirms that congestion is actually less likely in clear weather than in cloudy conditions. 

Overall, weather appears to have only a minor influence on congestion compared to other factors such as time of day, commuter patterns, and traffic demand.

*/
