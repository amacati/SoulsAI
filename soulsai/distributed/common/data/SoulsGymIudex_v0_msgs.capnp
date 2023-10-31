@0xb4dd05902eca5452;

struct DQNSample {
  obs @0 :List(Float32);
  action @1 :Int32;
  reward @2 :Float32;
  nextObs @3 :List(Float32);
  terminated @4 :Bool;
  truncated @5 :Bool;
  info @6 :Info;
  epSteps @7 :Int32;
  modelId @8 :Text;
}

struct Info {
    allowedActions @0: List(Int32);
}

struct EpisodeInfo {
  epReward @0 :Float32;
  epSteps @1 :Int32;
  bossHp @2 :Float32;
  win @3 :Bool;
  eps @4 :Float32;
  modelId @5 :Text;
}

struct Telemetry {
  epReward @0 :Float32;
  epSteps @1 :Int32;
  totalSteps @2 :Int32;
  bossHp @3 :Float32;
  win @4 :Bool;
  eps @5 :Float32;
}