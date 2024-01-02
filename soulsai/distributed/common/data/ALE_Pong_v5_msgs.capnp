@0x9f075f591ea0d034;

struct DQNSample {
  obs @0 :Data;
  obsShape @1 :List(Int64);
  action @2 :Int32;
  reward @3 :Float32;
  nextObs @4 :Data;
  terminated @5 :Bool;
  truncated @6 :Bool;
  info @7 :Info;
  modelId @8 :Text;
}

struct Info {
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
