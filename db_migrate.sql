create table if not exists cvr_prediction.cvr_predictions
(
    id         int auto_increment
        primary key,
    status     varchar(20)                         null,
    filename   varchar(255)                        not null,
    created_at timestamp default CURRENT_TIMESTAMP null
);

create table if not exists cvr_prediction.progress_logs
(
    id            int auto_increment
        primary key,
    prediction_id int                                 null,
    message       text                                null,
    timestamp     timestamp default CURRENT_TIMESTAMP null,
    constraint progress_logs_ibfk_1
        foreign key (prediction_id) references cvr_prediction.cvr_predictions (id)
);

create index prediction_id
    on cvr_prediction.progress_logs (prediction_id);