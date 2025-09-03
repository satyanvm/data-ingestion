CREATE TABLE IF NOT EXISTS public.argo_measurements
(
    id integer NOT NULL DEFAULT nextval('argo_measurements_id_seq'::regclass),
    platform_id character varying(20) COLLATE pg_catalog."default" NOT NULL,
    measurement_date timestamp with time zone,
    latitude real,
    longitude real,
    pressure_dbar real,
    temperature_celsius real,
    salinity_psu real,
    CONSTRAINT argo_measurements_pkey PRIMARY KEY (id)
)

TABLESPACE pg_default;

ALTER TABLE public.argo_measurements
    OWNER to postgres;
