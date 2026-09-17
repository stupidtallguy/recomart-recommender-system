CREATE SCHEMA IF NOT EXISTS recomart;
CREATE TABLE recomart.schema_migrations (
    version integer PRIMARY KEY,
    checksum text NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE recomart.snapshots (
    snapshot_id text PRIMARY KEY CHECK (snapshot_id ~ '^[0-9a-f]{64}$'),
    model_version text NOT NULL,
    snapshot_created_at timestamptz NOT NULL,
    known_users integer NOT NULL CHECK (known_users > 0),
    catalog_products integer NOT NULL CHECK (catalog_products > 0),
    popularity integer[] NOT NULL,
    imported_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE recomart.products (
    snapshot_id text NOT NULL REFERENCES recomart.snapshots(snapshot_id),
    product_id integer NOT NULL CHECK (product_id > 0),
    product_name text NOT NULL,
    aisle_id integer NOT NULL,
    aisle text NOT NULL,
    department_id integer NOT NULL,
    department text NOT NULL,
    PRIMARY KEY (snapshot_id, product_id)
);
CREATE TABLE recomart.user_recommendations (
    snapshot_id text NOT NULL REFERENCES recomart.snapshots(snapshot_id),
    user_id bigint NOT NULL CHECK (user_id > 0),
    history_orders integer NOT NULL CHECK (history_orders > 0),
    ranked integer[],
    repeat integer[],
    PRIMARY KEY (snapshot_id, user_id),
    CHECK (ranked IS NULL OR cardinality(ranked) <= 10),
    CHECK (repeat IS NULL OR cardinality(repeat) <= 10)
);
