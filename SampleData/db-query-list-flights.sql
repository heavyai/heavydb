USER admin omnisci {

DROP TABLE IF EXISTS flights;

CREATE TEMPORARY TABLE flights(
    Year_flight smallint,
    Month_flight smallint,
    DayofMonth smallint,
    DayOfWeek smallint,
    DepTime smallint,
    CRSDepTime smallint,
    ArrTime smallint,
    CRSArrTime smallint,
    UniqueCarrier text encoding dict,
    FlightNum smallint,
    TailNum text encoding dict,
    ActualElapsedTime smallint,
    CRSElapsedTime smallint,
    AirTime smallint,
    ArrDelay smallint,
    DepDelay smallint,
    Origin text encoding dict,
    Dest text encoding dict,
    Distance smallint,
    TaxiIn smallint,
    TaxiOut smallint,
    Cancelled smallint,
    CancellationCode text encoding dict,
    Diverted smallint,
    CarrierDelay smallint,
    WeatherDelay smallint,
    NASDelay smallint,
    SecurityDelay smallint,
    LateAircraftDelay smallint)
    FROM "CSV:SampleData/100_flights.csv";

    select count(*) from flights where LateAircraftDelay >= 10;
}
