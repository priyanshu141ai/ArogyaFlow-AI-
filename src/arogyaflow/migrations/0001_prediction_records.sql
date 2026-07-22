create table prediction_records (
    id bigint generated always as identity primary key,
    request_id text not null,
    prediction_type text not null,
    model_version text not null,
    schema_version text not null,
    request_payload jsonb not null,
    response_payload jsonb not null,
    created_at timestamptz not null,
    constraint prediction_records_type_check check (
        prediction_type in ('wait_time', 'arrivals', 'no_show', 'occupancy')
    ),
    constraint prediction_records_request_json_check check (
        jsonb_typeof(request_payload) = 'object'
    ),
    constraint prediction_records_response_json_check check (
        jsonb_typeof(response_payload) = 'object'
    )
);

create index prediction_records_type_created_at_idx
    on prediction_records (prediction_type, created_at desc);

create index prediction_records_request_id_idx
    on prediction_records (request_id);
