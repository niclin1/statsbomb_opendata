# Exemplos das Bases de Dados da StatsBomb

Abaixo estão exemplos do formato bruto dos dados extraídos da API da StatsBomb para ilustrar ao orientador a granularidade da base original que alimentou o nosso ETL.

## 1. Competições (`sb.competitions()`)
Retorna quais campeonatos e temporadas estão disponíveis na base aberta. Usamos isso para mapear os IDs necessários para baixar os jogos.
```json
{
  "competition_id": 9,
  "season_id": 281,
  "country_name": "Germany",
  "competition_name": "1. Bundesliga",
  "competition_gender": "male",
  "competition_youth": false,
  "competition_international": false,
  "season_name": "2023/2024",
  "match_updated": "2024-09-28T20:46:38.893391",
  "match_updated_360": "2025-07-06T04:26:07.636270",
  "match_available_360": "2025-07-06T04:26:07.636270",
  "match_available": "2024-09-28T20:46:38.893391"
}
```

## 2. Partidas (`sb.matches(competition_id, season_id)`)
Retorna a lista de jogos de um campeonato específico. Contém metadados importantes de cada partida (data, placar, estádio, juiz, técnicos).
```json
{
  "match_id": 3857256,
  "match_date": "2022-12-02",
  "kick_off": "21:00:00.000",
  "competition": "International - FIFA World Cup",
  "season": "2022",
  "home_team": "Serbia",
  "away_team": "Switzerland",
  "home_score": 2,
  "away_score": 3,
  "match_status": "available",
  "match_status_360": "available",
  "last_updated": "2023-02-17T23:45:15.306706",
  "last_updated_360": "2023-04-26T23:49:58.956186",
  "match_week": 3,
  "competition_stage": "Group Stage",
  "stadium": "Stadium 974",
  "referee": "Fernando Andr\u00e9s Rapallini",
  "home_managers": "Dragan Stojkovi\u0107",
  "away_managers": "Murat Yakin",
  "data_version": "1.1.0",
  "shot_fidelity_version": "2",
  "xy_fidelity_version": "2"
}
```

## 3. Escalações (`sb.lineups(match_id)`)
Retorna a ficha de cada jogador relacionado para a partida (titulares e reservas), incluindo posições nativas e país.
```json
{
  "player_id": 3500,
  "player_name": "Granit Xhaka",
  "player_nickname": null,
  "jersey_number": 10,
  "country": "Switzerland",
  "cards": [
    {
      "time": "94:21",
      "card_type": "Yellow Card",
      "reason": "Bad Behaviour",
      "period": 2
    }
  ],
  "positions": [
    {
      "position_id": 11,
      "position": "Left Defensive Midfield",
      "from": "00:00",
      "to": null,
      "from_period": 1,
      "to_period": null,
      "start_reason": "Starting XI",
      "end_reason": "Final Whistle"
    }
  ]
}
```

## 4. Eventos (`sb.events(match_id)`)
A base mais rica do projeto. Contém o log de tudo que aconteceu em campo ('Pass', 'Shot', 'Carry', 'Pressure', etc), junto com as coordenadas (x, y) de onde ocorreu. Foi daqui que extraímos o xG, Field Tilt e Zone 14.
```json
{
  "duration": 1.220932,
  "id": "4acb4fd2-f46a-4d73-993c-e06597873924",
  "index": 5,
  "location": [
    61.0,
    40.1
  ],
  "match_id": 3857256,
  "minute": 0,
  "pass_angle": 2.9431229,
  "pass_body_part": "Right Foot",
  "pass_end_location": [
    43.1,
    43.7
  ],
  "pass_height": "Ground Pass",
  "pass_length": 18.258423,
  "pass_recipient": "Remo Freuler",
  "pass_recipient_id": 6983.0,
  "pass_type": "Kick Off",
  "period": 1,
  "play_pattern": "From Kick Off",
  "player": "Breel-Donald Embolo",
  "player_id": 5545.0,
  "position": "Center Forward",
  "possession": 2,
  "possession_team": "Switzerland",
  "possession_team_id": 773,
  "related_events": [
    "b0d0583a-d544-4ab1-9043-6b42a57c27a8"
  ],
  "second": 0,
  "team": "Switzerland",
  "team_id": 773,
  "timestamp": "00:00:00.521",
  "type": "Pass"
}
```
